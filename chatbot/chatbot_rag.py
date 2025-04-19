import os
import sys
import warnings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from django.conf import settings
from openai import AzureOpenAI
from typing import List, Dict, Optional
import threading
from functools import lru_cache

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Cache for embeddings and responses
RESPONSE_CACHE = {}
EMBEDDING_CACHE = {}

# Global clients
client = None
embeddings = None
vectorstore = None

def initialize_clients():
    """Initialize clients once to avoid repeated initialization"""
    global client, embeddings, vectorstore
    
    if client is None:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )
    
    if embeddings is None:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY
        )
    
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"),
            embedding_function=embeddings,
            collection_name="goodwish_chatbot"
        )
    
    return client, embeddings, vectorstore

# LRU cache for embeddings
@lru_cache(maxsize=100)
def get_cached_embedding(query):
    """Cache embeddings for common queries"""
    client, embeddings, _ = initialize_clients()
    return embeddings.embed_query(query)

# Background task for non-critical operations
def background_task(fn):
    """Run function in background thread"""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
    return wrapper

@background_task
def log_token_usage(query, response):
    """Log token usage in background"""
    # Your token logging code here
    pass

def get_chatbot_response(query: str, image_data: Optional[str] = None, chat_history: List[Dict] = None) -> str:
    """
    Generate a chatbot response using RAG with Azure OpenAI and ChromaDB, supporting text and image inputs.
    
    Args:
        query: User's input text query (can be empty if image is provided).
        image_data: Base64-encoded image string (optional).
        chat_history: List of previous messages [{'role': 'user', 'content': str, 'image': str}, ...].
    
    Returns:
        Response string from the chatbot.
    """
    try:
        # Check if valid input is provided
        if not query and not image_data:
            return "Please provide a text query or an image."
            
        # Generate cache key based on query, image presence, and recent history
        history_suffix = ""
        if chat_history:
            # Use last 2 messages for cache key to keep it reasonably sized
            history_suffix = "_" + "_".join([f"{msg['role']}:{msg['content'][:20]}" for msg in chat_history[-2:]])
        
        image_suffix = "_with_image" if image_data else ""
        cache_key = f"{query[:50]}{image_suffix}{history_suffix}"
        
        # Check cache first
        if cache_key in RESPONSE_CACHE:
            return RESPONSE_CACHE[cache_key]

        # Initialize clients (uses globals to avoid repeated initialization)
        client, embeddings, vectorstore = initialize_clients()

        # Create retriever with more efficient parameters
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3
        
        # Retrieve context documents for relevant query
        search_query = query if query else "Describe the provided image"
        context_docs = retriever.invoke(search_query)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Format chat history for prompt (limiting to just last 3 messages for efficiency)
        formatted_history = ""
        if chat_history:
            for msg in chat_history[-3:]:  # Reduced from 5 to 3
                content = msg['content']
                if msg.get('image'):
                    content += " [Image provided]"
                formatted_history += f"{msg['role']}: {content}\n"
        
        # Prepare system prompt with fewer constraints
        system_prompt = f"""
        ### Role
        - You are an AI chatbot designed to assist users with helpful and informative responses. You can handle greetings, small talk, and specific queries.

        ### Capabilities
        1. Language: Respond in English if the query is in English. If the query is in Nepali or Romanized Nepali, respond in pure Nepali (never Romanized Nepali).
        2. Knowledge: Use the provided context when relevant, but you can also draw on your general knowledge to provide helpful responses.
        3. Word Limit: Keep responses between 30-50 words.
        4. Greetings: Respond to greetings like "hello" or "नमस्ते" with a friendly reply.
        5. If there links provided when fetched from RAG always show that link no matter what.
        6. When talking about wishchat always provide the link cleverly asking user to navigate there. If link is not provided in response of RAG use this: https://wishchat.goodwish.com.np
        7. Also give them the goodwish engineering socials when asked if not fetched from RAG.  - Facebook: https://www.facebook.com/Goodwish-Engineering-61571584179109/
  - LinkedIn: https://www.linkedin.com/company/goodwish-engineering/posts/?feedView=all


        Chat history:
        {formatted_history}

        Context:
        {context}
        """

        # Prepare chat messages
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            }
        ]
        
        # Add user message with text and/or image
        user_content = []
        if query:
            user_content.append({"type": "text", "text": query})
        if image_data:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        # Append user message (default to image description if no query)
        messages.append({
            "role": "user",
            "content": user_content or [{"type": "text", "text": "Describe the provided image"}]
        })
        
        # Generate completion with optimized parameters
        completion = client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=200,  # Reduced from 800 to 200
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        # Extract response
        response = completion.choices[0].message.content
        
        # Remove any markdown formatting that might appear
        response = response.replace("###", "").replace("##", "").replace("#", "")
        response = response.replace("***", "").replace("**", "").replace("*", "")
        response = response.replace('(',"").replace(')',"")
        
        # Cache the response
        RESPONSE_CACHE[cache_key] = response
        
        # Log token usage in background
        log_token_usage(query, response)
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't process that. Try WishChat Enterprise at info@goodwish.com.np!"

if __name__ == "__main__":
    # Add project directory to sys.path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_dir)
    
    # Set Django settings module
    os.environ["DJANGO_SETTINGS_MODULE"] = "goodwish_chatbot.settings"
    import django
    django.setup()

    # Initialize clients on startup
    initialize_clients()

    # Interactive loop
    print("Welcome to Goodwish Chatbot! Type 'quit' to exit.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        response = get_chatbot_response(query, None, chat_history)
        print(f"Bot: {response}")
        # Update history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
        chat_history = chat_history[-6:]  # Limit to last 6 messages
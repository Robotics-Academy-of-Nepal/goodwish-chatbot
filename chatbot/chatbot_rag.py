import os
import sys
import warnings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from django.conf import settings
from openai import AzureOpenAI
from typing import List, Dict, Optional

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

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
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )

        # Initialize embeddings for RAG
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY
        )

        # Load ChromaDB vector store
        vectorstore = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"),
            embedding_function=embeddings,
            collection_name="goodwish_chatbot"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Retrieve context documents
        context_docs = retriever.invoke(query or "Describe the provided image")
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Format chat history for prompt
        formatted_history = ""
        if chat_history:
            for msg in chat_history[-5:]:
                content = msg['content']
                if msg.get('image'):
                    content += " [Image provided]"
                formatted_history += f"{msg['role']}: {content}\n"
        
        # Prepare system prompt
        system_prompt = f"""
 ### Role
        - You are an AI chatbot designed to assist users. You can also handle greetings and small talk politely.

        ### Capabilities
        1. Language: Respond in English if the query is in English. If the query is in Nepali or Romanized Nepali, respond in pure Nepali (never Romanized Nepali).
        2. Scope: If the query is unrelated to the document content, analyze the retrieved context and creatively redirect. Say something like: "I’m sorry, I don’t have info on your query, but I have knowledge about [document topic, e.g., robotics]. How can I assist with that?" (English) or "माफ गर्नुहोस्, मसँग तपाईंको प्रश्नको जानकारी छैन, तर म [document topic, e.g., रोबोटिक्स] बारे जानकार छु। त्यसमा कसरी मद्दत गर्न सक्छु?" (Nepali).
        3. Word Limit: Keep responses between 80-100 words.
        4. Greetings: Respond to greetings like "hello" or "नमस्ते" with a friendly reply, e.g., "Hello! How can I assist you today?" or "नमस्ते! म तपाईंलाई आज कसरी सहयोग गर्न सक्छु?"

        ### Constraints
        - Base answers on document content unless it’s a greeting or small talk. Use the context to infer the document topic for off-topic responses.
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
        
        # Generate completion
        completion = client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        # Extract response
        response = completion.choices[0].message.content
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

    # Interactive loop
    print("Welcome to Goodwish Chatbot! Type 'quit' to exit.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        response = get_chatbot_response(query, chat_history)
        print(f"Bot: {response}")
        # Update history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
        chat_history = chat_history[-10:]  # Limit to last 10 messages
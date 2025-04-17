import os
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.conf import settings

def ingest_documents():
    # Define document paths
    document_dir = os.path.join(os.path.dirname(__file__), "docs")
    document_files = [
        "Goodwish_Engineering_Company_Information.txt",
        "WishChat_Product_Information.txt"
    ]

    # Load documents
    documents = []
    for file_name in document_files:
        file_path = os.path.join(document_dir, file_name)
        if os.path.exists(file_path):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        else:
            print(f"Warning: {file_path} not found")

    if not documents:
        print("Error: No documents were loaded. Check file paths.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
    )

    # Initialize ChromaDB client
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists
    collections = client.list_collections()
    for collection in collections:
        if collection.name == "goodwish_chatbot":
            print("Dropping existing 'goodwish_chatbot' collection...")
            client.delete_collection("goodwish_chatbot")
            break
    
    # Create or update ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,  # Pass the client directly
        collection_name="goodwish_chatbot"
    )
    # No need to call persist() - it's handled automatically
    
    print(f"Embedded {len(chunks)} document chunks into ChromaDB")

if __name__ == "__main__":
    # For standalone execution, set up Django environment
    import sys
    import django
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_dir)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "goodwish_chatbot.settings")
    django.setup()
    
    ingest_documents()
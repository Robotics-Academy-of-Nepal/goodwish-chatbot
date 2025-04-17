from django.core.management.base import BaseCommand
from chatbot.document_ingestion import ingest_documents

class Command(BaseCommand):
    help = "Ingest documents into ChromaDB vector store"

    def handle(self, *args, **options):
        self.stdout.write("Starting document ingestion...")
        ingest_documents()
        self.stdout.write(self.style.SUCCESS("Document ingestion completed"))
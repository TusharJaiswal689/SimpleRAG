from app.core.config import settings

from app.embedder.embedding import Embedder
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.indexer import Indexer
from app.retrieval.retriever import Retriever
from app.llm.client import OllamaClient
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


def get_embedder():
    return Embedder(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_EMBEDDING_MODEL
    )


def get_ollama_client():
    return OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_GENERATION_MODEL
    )


def get_retriever():
    return Retriever(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME,
        embedder=get_embedder(),
        top_k=settings.TOP_K
    )


def get_rag_service():
    return RAGService(
        retriever=get_retriever(),
        ollama_client=get_ollama_client()
    )


def get_ingestion_service():
    loader = DocumentLoader()

    chunker = DocumentChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    embedder = get_embedder()

    indexer = Indexer(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME
    )

    return IngestionService(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        indexer=indexer
    )
from app.core.config import settings
from app.embedder.embedding import Embedder
from pinecone import Pinecone


embedder = Embedder(
    settings.OLLAMA_BASE_URL,
    settings.OLLAMA_EMBEDDING_MODEL
)

pc = Pinecone(
    api_key=settings.PINECONE_API_KEY
)

index = pc.Index(
    settings.PINECONE_INDEX_NAME
)


text = "This is a test document about database normalization."

vector = embedder.embed_text(text)


index.upsert(
    vectors=[
        {
            "id": "test-1",
            "values": vector,
            "metadata": {
                "text": text,
                "source": "test"
            }
        }
    ]
)

print("Vector inserted successfully.")
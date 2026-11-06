from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding


class Embedder:
    """Standardized Ollama embedding wrapper used by ingestion and retrieval."""

    def __init__(self, base_url: str, model: str, expected_dimension: int | None = None):
        self.base_url = base_url
        self.model = model
        self.expected_dimension = expected_dimension
        self.embedding = OllamaEmbedding(
            model_name=model,
            base_url=base_url,
        )

    def _validate_vector(self, vector: list[float], context: str) -> list[float]:
        if self.expected_dimension is not None and len(vector) != self.expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model}: expected {self.expected_dimension}, got {len(vector)}. "
                f"Verify the Ollama embedding model and Pinecone vector dimension before indexing or querying."
            )
        return vector

    def get_dimension(self) -> int:
        vector = self.embedding.get_text_embedding("dimension check")
        return len(self._validate_vector(vector, "dimension check"))

    def embed_text(self, text: str) -> list[float]:
        return self._validate_vector(
            self.embedding.get_text_embedding(text),
            "single text embedding",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.embedding.get_text_embedding_batch(texts)
        for vector in vectors:
            self._validate_vector(vector, "document embedding batch")
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)
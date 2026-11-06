import logging
from time import perf_counter

from pinecone import Pinecone


logger = logging.getLogger(__name__)


class Retriever:
    """Cosine-score retrieval; threshold is a starting point, not an entailment test."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedder,
        top_k: int = 5,
        namespace: str | None = None,
        min_relevance_score: float = 0.70,
    ):
        if not -1.0 <= min_relevance_score <= 1.0:
            raise ValueError("Cosine relevance threshold must be between -1 and 1.")

        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.embedder = embedder
        self.top_k = top_k
        self.namespace = namespace
        self.min_relevance_score = min_relevance_score

    def retrieve(self, query: str) -> list[dict]:
        started = perf_counter()
        try:
            query_vector = self.embedder.embed_query(query)
        finally:
            logger.info("RAG timing | embedding_ms=%.1f", (perf_counter() - started) * 1000)

        params = {
            "vector": query_vector,
            "top_k": self.top_k,
            "include_metadata": True,
        }
        if self.namespace is not None:
            params["namespace"] = self.namespace

        started = perf_counter()
        try:
            response = self.index.query(**params)
        finally:
            logger.info(
                "RAG timing | pinecone_retrieval_ms=%.1f",
                (perf_counter() - started) * 1000,
            )

        results = []
        for match in self._field(response, "matches", []) or []:
            score = self._field(match, "score")
            metadata = self._field(match, "metadata") or {}
            if score is None or score < self.min_relevance_score:
                continue

            result = dict(metadata)
            result["_score"] = float(score)
            results.append(result)
        return results

    @staticmethod
    def _field(value, name, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)
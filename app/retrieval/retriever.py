from pinecone import Pinecone


class Retriever:

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedder,
        top_k: int = 5
    ):
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_vector = self.embedder.embed_query(query)

        response = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        return [
            match["metadata"]
            for match in response["matches"]
        ]
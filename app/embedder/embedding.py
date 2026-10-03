from ollama import Client


class Embedder:

    def __init__(self, base_url: str, model: str):
        self.client = Client(host=base_url)
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embed(
            model=self.model,
            input=text
        )

        return response["embeddings"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            model=self.model,
            input=texts
        )

        return response["embeddings"]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)
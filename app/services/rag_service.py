from app.prompts.rag_prompt import build_rag_prompt


class RAGService:

    def __init__(self, retriever, ollama_client):
        self.retriever = retriever
        self.ollama_client = ollama_client

    def generate_answer(
        self,
        query: str,
        history: list | None = None
    ):
        history = history or []

        contexts = self.retriever.retrieve(query)

        prompt = build_rag_prompt(
            query=query,
            contexts=contexts,
            history=history
        )

        return self.ollama_client.generate_stream(prompt)
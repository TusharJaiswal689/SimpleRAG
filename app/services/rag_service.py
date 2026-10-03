from app.prompts.rag_prompt import build_rag_prompt


class RAGService:

    def __init__(
        self,
        retriever,
        ollama_client,
        query_rewriter
    ):
        self.retriever = retriever
        self.ollama_client = ollama_client
        self.query_rewriter = query_rewriter

    def generate_answer(
        self,
        query: str,
        history: list | None = None
    ):
        history = history or []

        retrieval_query = self.query_rewriter.rewrite(
            query=query,
            history=history
        )

        contexts = self.retriever.retrieve(
            retrieval_query
        )

        prompt = build_rag_prompt(
            query=query,
            contexts=contexts,
            history=history
        )

        return self.ollama_client.generate_stream(prompt)
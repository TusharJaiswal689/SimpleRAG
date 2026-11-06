import logging
from time import perf_counter

from app.prompts.rag_prompt import build_rag_prompt
from app.routing.query_router import QueryRouter

logger = logging.getLogger(__name__)


class RAGService:

    def __init__(self, retriever, ollama_client, query_rewriter, query_router: QueryRouter | None = None, llama_runtime=None):
        self.retriever = retriever
        self.ollama_client = ollama_client
        self.query_rewriter = query_rewriter
        self.query_router = query_router or QueryRouter()
        self.llama_runtime = llama_runtime

    def generate_answer(self, query: str, history: list | None = None):
        for event in self.generate_events(query, history):
            if event["event"] == "token":
                yield event["data"]

    def generate_events(self, query: str, history: list | None = None):
        started = perf_counter()
        try:
            yield from self._generate_events(query, history)
        finally:
            logger.info(
                "RAG timing | total_response_ms=%.1f",
                (perf_counter() - started) * 1000,
            )

    def _generate_events(self, query: str, history: list | None = None):
        history = history or []
        decision = self.query_router.classify(query, history)

        logger.info(
            "RAG routing | route=%s | "
            "should_retrieve=%s | history_count=%d | roles=%s",
            decision.route,
            decision.should_retrieve,
            len(history),
            [msg["role"] for msg in history if isinstance(msg, dict) and "role" in msg],
        )

        if decision.route == "conversation":
            yield self._token_event(
                "I can help with your notes and prior context. Ask a question about the uploaded material and I will use it to answer."
            )
            return

        if decision.route == "contextual":
            for token in self._generate_contextual_response(query, history):
                yield self._token_event(token)
            return

        if decision.route == "guardrail":
            yield self._token_event(
                "Please provide a clear question or request so I can help with the uploaded document context."
            )
            return

        if decision.route == "clarification":
            yield self._token_event(
                "I need a bit more context before I can answer accurately. Please specify what you want to know."
            )
            return

        rewrite_started = perf_counter()
        should_rewrite = self.query_rewriter.needs_rewrite(query, history)
        try:
            if should_rewrite:
                retrieval_query = self.query_rewriter.rewrite(
                    query=query,
                    history=history,
                )
            else:
                retrieval_query = query
        finally:
            logger.info(
                "RAG timing | query_rewrite_ms=%.1f | rewrite=%s",
                (perf_counter() - rewrite_started) * 1000,
                should_rewrite,
            )

        contexts = self.retriever.retrieve(retrieval_query)

        if not contexts:
            yield self._token_event(
                "I could not find sufficiently relevant context in the uploaded notes for that question."
            )
            return

        prompt = build_rag_prompt(
            query=query,
            contexts=contexts,
            history=history,
        )

        for token in self._stream_with_timing(prompt, "retrieval"):
            yield self._token_event(token)

        sources = self._source_payload(contexts)
        if sources:
            yield {"event": "sources", "data": sources}

    def _generate_contextual_response(self, query: str, history: list | None = None):
        history = history or []
        prior_assistant_answer = ""

        for message in reversed(history):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role == "assistant" and content:
                prior_assistant_answer = content
                break

        prompt = (
            "Use the previous assistant answer as the relevant context for the current request without consulting documents.\n"
            f"Previous assistant answer:\n{prior_assistant_answer or 'No prior answer is available. '}\n\n"
            f"Current request:\n{query}\n\nAnswer only from that prior context."
        )

        yield from self._stream_with_timing(prompt, "contextual")

    def _stream_with_timing(self, prompt: str, route: str):
        started = perf_counter()
        stream = (
            self.llama_runtime.generate_stream(prompt)
            if self.llama_runtime is not None
            else self.ollama_client.generate_stream(prompt)
        )
        first_token_logged = False

        for token in stream:
            if token and not first_token_logged:
                logger.info(
                    "RAG timing | time_to_first_token_ms=%.1f | route=%s",
                    (perf_counter() - started) * 1000,
                    route,
                )
                first_token_logged = True
            yield token

    @staticmethod
    def _token_event(token: str) -> dict:
        return {"event": "token", "data": token}

    @staticmethod
    def _source_payload(contexts: list[dict]) -> list[dict]:
        sources = []
        seen = set()
        for context in contexts:
            filename = context.get("filename")
            page_number = context.get("page_number", context.get("page_label"))
            chunk_id = context.get("chunk_id")
            key = (filename, page_number, chunk_id)
            if key in seen:
                continue
            seen.add(key)

            source = {}
            if filename:
                source["filename"] = filename
            if page_number is not None:
                source["page_number"] = page_number
            if source:
                sources.append(source)
        return sources

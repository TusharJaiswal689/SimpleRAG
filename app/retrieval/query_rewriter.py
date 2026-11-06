import re

from ollama import Client


class QueryRewriter:
    CONTEXT_REFERENCE_PATTERN = re.compile(
        r"\b(?:it|its|they|them|their|this|that|these|those|former|latter|"
        r"same|above|previous|earlier|mentioned|there)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        context_window: int,
    ):
        self.client = Client(host=base_url)
        self.model = model
        self.context_window = context_window

    def needs_rewrite(self, query: str, history: list | None = None) -> bool:
        return bool(history) and bool(self.CONTEXT_REFERENCE_PATTERN.search(query))

    def rewrite(
        self,
        query: str,
        history: list | None = None
    ) -> str:

        history = history or []

        if not history:
            return query

        history_text = "\n".join(
            f"{message.role}: {message.content}"
            if hasattr(message, "role")
            else f"{message['role']}: {message['content']}"
            for message in history
        )

        prompt = f"""
Rewrite the user's current question into a standalone search query
for retrieving relevant information from college notes.

Use the conversation history to resolve references such as:
"it", "this", "that", "they", or "those".

Do not answer the question.
Return ONLY the rewritten search query.

Conversation history:
{history_text}

Current question:
{query}

Search query:
""".strip()

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            think=False,
            options={"num_ctx": self.context_window},
        )

        rewritten_query = response["response"].strip()

        return rewritten_query or query
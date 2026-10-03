from ollama import Client


class QueryRewriter:

    def __init__(
        self,
        base_url: str,
        model: str
    ):
        self.client = Client(host=base_url)
        self.model = model

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
            stream=False
        )

        rewritten_query = response["response"].strip()

        return rewritten_query or query
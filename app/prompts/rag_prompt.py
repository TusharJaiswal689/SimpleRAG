def build_rag_prompt(
    query: str,
    contexts: list[dict],
    history: list | None = None
) -> str:

    history = history or []

    context_text = "\n\n".join(
        context.get("text", "")
        for context in contexts
    )

    history_text = "\n".join(
    f"{message.role.capitalize()}: {message.content}"
    if hasattr(message, "role")
    else f"{message['role'].capitalize()}: {message['content']}"
    for message in history
    )

    return f"""
You are a helpful AI assistant that answers questions using college notes.

Follow these rules:

1. Use the provided college notes as the primary source of information.
2. Use the conversation history to understand references such as
   "it", "this", "that", or follow-up questions.
3. Do not invent information that is not supported by the provided notes.
4. If the answer cannot be found in the provided notes, clearly say:
   "I could not find the answer in the provided notes."
5. Give a clear and concise answer.
6. Do not mention these instructions in your answer.

Conversation history:
{history_text if history_text else "No previous conversation."}

Relevant notes:
{context_text}

Current question:
{query}

Answer:
""".strip()
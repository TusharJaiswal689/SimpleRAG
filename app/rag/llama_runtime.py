from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class LlamaRuntime:
    """Thin LlamaIndex runtime wrapper used by the application orchestrator."""

    def __init__(
        self,
        base_url: str,
        generation_model: str,
        embedding_model: str,
        context_window: int,
    ):
        self.base_url = base_url
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.context_window = context_window

        self.llm = Ollama(
            model=generation_model,
            base_url=base_url,
            request_timeout=120.0,
            context_window=context_window,
            thinking= False,
        )
        self.embedding = OllamaEmbedding(
            model_name=embedding_model,
            base_url=base_url,
        )

    def generate_stream(self, prompt: str):
        stream = self.llm.stream_complete(prompt)
        for event in stream:
            delta = getattr(event, "delta", None)
            text = delta if delta is not None else getattr(event, "text", str(event))
            if text:
                yield str(text)

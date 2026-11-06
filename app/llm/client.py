from collections.abc import Iterator

from ollama import Client


class OllamaClient:

    def __init__(self, base_url: str, model: str, context_window: int):
        self.client = Client(host=base_url)
        self.model = model
        self.context_window = context_window

    def generate_stream(self, prompt: str) -> Iterator[str]:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            think=False,
            options={"num_ctx": self.context_window},
        )

        for chunk in response:
            yield chunk["response"]
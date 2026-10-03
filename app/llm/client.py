from collections.abc import Iterator

from ollama import Client


class OllamaClient:

    def __init__(self, base_url: str, model: str):
        self.client = Client(host=base_url)
        self.model = model

    def generate_stream(self, prompt: str) -> Iterator[str]:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True
        )

        for chunk in response:
            yield chunk["response"]
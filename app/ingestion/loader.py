from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader


class DocumentLoader:

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            reader = PDFReader()
            return reader.load_data(file=path)

        if path.suffix.lower() == ".docx":
            reader = DocxReader()
            return reader.load_data(file=path)

        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )
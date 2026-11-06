from pathlib import Path
import hashlib
import unicodedata

from llama_index.core import Document
from llama_index.readers.file import PDFReader, DocxReader


class DocumentLoader:

    def load(self, file_path: str, filename: str | None = None) -> list[Document]:
        path = Path(file_path)
        source_filename = filename or path.name

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document not found: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Document is empty: {path.name}")

        if path.suffix.lower() == ".pdf":
            reader = PDFReader()
            documents = reader.load_data(file=path)
        elif path.suffix.lower() == ".docx":
            reader = DocxReader()
            documents = reader.load_data(file=path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if not documents:
            raise ValueError(f"No readable content found in document: {path.name}")

        for document in documents:
            metadata = dict(getattr(document, "metadata", {}) or {})
            metadata["filename"] = source_filename
            metadata.setdefault("source_path", source_filename)
            if "page_number" not in metadata:
                page_label = metadata.get("page_label")
                if page_label is not None:
                    try:
                        metadata["page_number"] = int(page_label)
                    except (TypeError, ValueError):
                        metadata["page_number"] = str(page_label)
            normalized_name = unicodedata.normalize("NFC", source_filename).casefold()
            metadata["document_id"] = hashlib.sha256(
                normalized_name.encode("utf-8")
            ).hexdigest()
            document.metadata = metadata

        return documents
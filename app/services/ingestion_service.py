from pathlib import Path


class IngestionService:

    def __init__(self, loader, chunker, embedder, indexer):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer

    def ingest(self, file_path: str, filename: str | None = None):
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document not found: {path.name}")

        source_filename = filename or path.name
        documents = self.loader.load(str(path), filename=source_filename)
        chunks = self.chunker.chunk(documents)
        if not chunks:
            raise ValueError(f"No chunks were generated for {source_filename}.")

        texts = [chunk.get_content().strip() for chunk in chunks]
        if not any(texts):
            raise ValueError(f"No readable content was found in {source_filename}.")

        vectors = self.embedder.embed_documents(texts)
        indexed_count = self.indexer.index_chunks(
            chunks=chunks,
            vectors=vectors,
        )

        return {
            "filename": source_filename,
            "chunks_indexed": indexed_count,
            "namespace": getattr(self.indexer, "namespace", None),
        }
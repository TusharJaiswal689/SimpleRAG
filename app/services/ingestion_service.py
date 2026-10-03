from pathlib import Path


class IngestionService:

    def __init__(self, loader, chunker, embedder, indexer):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer

    def ingest(self, file_path: str):

        documents = self.loader.load(file_path)

        chunks = self.chunker.chunk(documents)

        texts = [
            chunk.get_content()
            for chunk in chunks
        ]

        vectors = self.embedder.embed_documents(texts)

        self.indexer.index_chunks(
            chunks=chunks,
            vectors=vectors
        )

        return {
            "filename": Path(file_path).name,
            "chunks_indexed": len(chunks)
        }
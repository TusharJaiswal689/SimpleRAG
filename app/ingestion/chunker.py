from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentChunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: list[Document]):
        return self.splitter.get_nodes_from_documents(documents)
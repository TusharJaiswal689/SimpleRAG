from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentChunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]):
        nodes = self.splitter.get_nodes_from_documents(documents)

        for node_index, node in enumerate(nodes):
            metadata = dict(getattr(node, "metadata", {}) or {})
            document_id = metadata.get("document_id") or metadata.get("filename") or f"document-{node_index}"
            metadata.setdefault("document_id", document_id)
            metadata["chunk_index"] = node_index
            metadata.setdefault("chunk_id", node.node_id)
            node.metadata = metadata

        return nodes
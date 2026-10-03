from pinecone import Pinecone


class Indexer:

    def __init__(self, api_key: str, index_name: str):
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)

    def index_chunks(self, chunks, vectors):
        records = []

        for chunk, vector in zip(chunks, vectors):
            records.append({
                "id": chunk.node_id,
                "values": vector,
                "metadata": {
                    "text": chunk.get_content(),
                    **chunk.metadata
                }
            })

        self.index.upsert(vectors=records)
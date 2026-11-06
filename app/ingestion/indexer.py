from __future__ import annotations

import hashlib
import unicodedata

from pinecone import Pinecone


class Indexer:

    def __init__(self, api_key: str, index_name: str, namespace: str | None = None):
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.namespace = namespace

    def _build_record_id(self, chunk, index: int) -> str:
        metadata = getattr(chunk, "metadata", {}) or {}
        document_id = (
            metadata.get("document_id")
            or metadata.get("filename")
            or "unknown-document"
        )
        text = (chunk.get_content() or "").strip()
        document_key = hashlib.sha256(str(document_id).encode("utf-8")).hexdigest()[:16]
        content_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{document_key}:{index}:{content_key}"

    def index_chunks(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("Each document chunk must have exactly one embedding vector.")

        records = []
        seen: set[str] = set()

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            record_id = self._build_record_id(chunk, index)
            if record_id in seen:
                continue
            seen.add(record_id)

            metadata = dict(getattr(chunk, "metadata", {}) or {})
            if not metadata.get("document_id"):
                metadata["document_id"] = metadata.get("filename") or "unknown-document"
            metadata.setdefault("chunk_index", index)
            metadata["text"] = chunk.get_content()
            metadata["chunk_id"] = record_id

            record = {
                "id": record_id,
                "values": vector,
                "metadata": metadata,
            }
            records.append(record)

        if not records:
            return 0

        payload = {"vectors": records}
        if self.namespace is not None:
            payload["namespace"] = self.namespace

        if self.namespace is None:
            raise ValueError(
                "A Pinecone namespace must be configured to safely replace a document."
            )

        # Upsert first; only remove stale chunks after Pinecone accepts the replacement.
        self.index.upsert(**payload)

        filenames = {
            record["metadata"].get("filename")
            for record in records
            if record["metadata"].get("filename")
        }
        if len(filenames) == 1:
            filename = next(iter(filenames))
            retained_ids = {record["id"] for record in records}
            stale_ids = self._find_stale_ids(filename, retained_ids)
            for start in range(0, len(stale_ids), 100):
                self.index.delete(
                    ids=stale_ids[start:start + 100],
                    namespace=self.namespace,
                )
        return len(records)

    def _find_stale_ids(self, filename: str, retained_ids: set[str]) -> list[str]:
        stale_ids = []
        for page in self.index.list(namespace=self.namespace, limit=100):
            page_ids = [
                vector.id
                for vector in page.vectors
                if vector.id not in retained_ids
            ]
            for start in range(0, len(page_ids), 100):
                batch = page_ids[start:start + 100]
                if not batch:
                    continue
                fetched = self.index.fetch(ids=batch, namespace=self.namespace)
                for vector_id, vector in fetched.vectors.items():
                    metadata = (
                        vector.get("metadata", {})
                        if isinstance(vector, dict)
                        else getattr(vector, "metadata", None)
                    ) or {}
                    old_filename = metadata.get("filename")
                    if (
                        old_filename
                        and unicodedata.normalize("NFC", old_filename).casefold()
                        == unicodedata.normalize("NFC", filename).casefold()
                    ):
                        stale_ids.append(vector_id)
        return stale_ids
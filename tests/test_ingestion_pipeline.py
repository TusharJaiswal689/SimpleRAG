from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.embedder.embedding import Embedder
from app.api.routes_documents import upload_document
from app.ingestion.chunker import DocumentChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.services.ingestion_service import IngestionService


def test_embedder_uses_standardized_ollama_embedding_and_validates_dimension():
    with patch("app.embedder.embedding.OllamaEmbedding") as mock_ollama_embedding:
        mock_embedding = Mock()
        mock_embedding.get_text_embedding.return_value = [0.0] * 768
        mock_embedding.get_text_embedding_batch.return_value = [[0.0] * 768, [0.0] * 768]
        mock_ollama_embedding.return_value = mock_embedding

        embedder = Embedder(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
            expected_dimension=768,
        )

        assert embedder.model == "nomic-embed-text"
        assert embedder.get_dimension() == 768
        assert len(embedder.embed_documents(["alpha", "beta"])) == 2
        assert embedder.embed_query("alpha") == [0.0] * 768


def test_indexer_uses_isolated_namespace_and_stable_chunk_ids():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        index_client.list.return_value = []
        index_client.list.return_value = []

        indexer = Indexer(
            api_key="test-key",
            index_name="college-notes",
            namespace="phase3-test",
        )

        chunks = [
            SimpleNamespace(
                metadata={"document_id": "doc-1", "filename": "note.pdf"},
                get_content=lambda: "alpha",
                node_id="node-1",
            ),
            SimpleNamespace(
                metadata={"document_id": "doc-1", "filename": "note.pdf"},
                get_content=lambda: "beta",
                node_id="node-2",
            ),
        ]
        vectors = [[1.0, 2.0], [3.0, 4.0]]

        indexer.index_chunks(chunks, vectors)

        index_client.upsert.assert_called_once()
        payload = index_client.upsert.call_args.kwargs
        assert payload["namespace"] == "phase3-test"
        assert len(payload["vectors"]) == 2
        assert payload["vectors"][0]["metadata"]["chunk_id"]


def test_retriever_queries_with_same_namespace_embedding_and_timing(caplog):
    embedder = Mock()
    embedder.embed_query.return_value = [0.0, 1.0]
    index_client = Mock()
    index_client.query.return_value = {
        "matches": [{"metadata": {"text": "hello"}, "score": 0.81}]
    }

    with patch("app.retrieval.retriever.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client

        retriever = Retriever(
            api_key="test-key",
            index_name="college-notes",
            embedder=embedder,
            top_k=2,
            namespace="phase3-test",
        )

        with caplog.at_level("INFO"):
            result = retriever.retrieve("what is this?")

        assert result == [{"text": "hello", "_score": 0.81}]
        embedder.embed_query.assert_called_once_with("what is this?")
        params = index_client.query.call_args.kwargs
        assert params["namespace"] == "phase3-test"
        assert params["top_k"] == 2
        timing_messages = [record.getMessage() for record in caplog.records]
        assert any("embedding_ms=" in message for message in timing_messages)
        assert any("pinecone_retrieval_ms=" in message for message in timing_messages)


def test_document_loader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        DocumentLoader().load(str(path))


def test_ingestion_service_returns_namespace_and_validates_chunk_output(tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"sample content")

    loader = Mock()
    loader.load.return_value = [SimpleNamespace(metadata={"filename": "sample.pdf"})]

    chunker = Mock()
    chunker.chunk.return_value = [
        SimpleNamespace(metadata={"document_id": "doc-1", "filename": "sample.pdf"}, get_content=lambda: "chunk one", node_id="chunk-1"),
        SimpleNamespace(metadata={"document_id": "doc-1", "filename": "sample.pdf"}, get_content=lambda: "chunk two", node_id="chunk-2"),
    ]

    embedder = Mock()
    embedder.embed_documents.return_value = [[0.0, 1.0], [1.0, 0.0]]

    indexer = Mock()
    indexer.namespace = "phase3-test"
    indexer.index_chunks.return_value = 2

    service = IngestionService(loader=loader, chunker=chunker, embedder=embedder, indexer=indexer)
    payload = service.ingest(str(file_path))

    assert payload["chunks_indexed"] == 2
    assert payload["namespace"] == "phase3-test"
    embedder.embed_documents.assert_called_once_with(["chunk one", "chunk two"])


def test_embedder_raises_on_dimension_mismatch():
    with patch("app.embedder.embedding.OllamaEmbedding") as mock_ollama_embedding:
        mock_embedding = Mock()
        mock_embedding.get_text_embedding.return_value = [0.0] * 255
        mock_ollama_embedding.return_value = mock_embedding

        embedder = Embedder(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
            expected_dimension=768,
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            embedder.embed_query("short")


def test_retriever_returns_empty_results():
    embedder = Mock()
    embedder.embed_query.return_value = [0.0, 0.0]
    index_client = Mock()
    index_client.query.return_value = {"matches": []}

    with patch("app.retrieval.retriever.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        retriever = Retriever(
            api_key="test-key",
            index_name="college-notes",
            embedder=embedder,
            top_k=5,
            namespace="phase3-test",
        )

        assert retriever.retrieve("irrelevant query") == []


def test_retriever_filters_low_relevance_matches_and_preserves_score():
    embedder = Mock()
    embedder.embed_query.return_value = [0.0, 0.0]
    index_client = Mock()
    index_client.query.return_value = {
        "matches": [
            {"metadata": {"text": "weak match", "filename": "note.pdf"}, "score": 0.2}
        ]
    }

    with patch("app.retrieval.retriever.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        retriever = Retriever(
            api_key="test-key",
            index_name="college-notes",
            embedder=embedder,
            top_k=5,
            namespace="phase3-test",
        )

        assert retriever.retrieve("fuzzy question") == []


def test_retriever_keeps_relevant_and_partially_relevant_results_only():
    embedder = Mock()
    embedder.embed_query.return_value = [0.0, 0.0]
    index_client = Mock()
    index_client.query.return_value = {
        "matches": [
            {
                "metadata": {"text": "supported passage", "filename": "notes.pdf"},
                "score": 0.87,
            },
            {
                "metadata": {"text": "unrelated passage", "filename": "other.pdf"},
                "score": 0.12,
            },
        ]
    }
    with patch("app.retrieval.retriever.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        retriever = Retriever(
            api_key="test-key",
            index_name="college-notes",
            embedder=embedder,
            namespace="phase3-test",
            min_relevance_score=0.35,
        )

        assert retriever.retrieve("supported question") == [
            {"text": "supported passage", "filename": "notes.pdf", "_score": 0.87}
        ]


def test_retriever_rejects_missing_or_invalid_score_threshold():
    with pytest.raises(ValueError, match="between -1 and 1"):
        Retriever(
            api_key="test-key",
            index_name="college-notes",
            embedder=Mock(),
            min_relevance_score=1.1,
        )

    embedder = Mock()
    embedder.embed_query.return_value = [0.0]
    index_client = Mock()
    index_client.query.return_value = {
        "matches": [{"metadata": {"text": "no score"}}]
    }
    with patch("app.retrieval.retriever.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        retriever = Retriever("test-key", "college-notes", embedder)
        assert retriever.retrieve("query") == []


def test_indexer_first_upload_upserts_without_deleting_other_vectors():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        index_client.list.return_value = []

        indexer = Indexer(
            api_key="test-key",
            index_name="college-notes",
            namespace="phase3-test",
        )

        chunk = SimpleNamespace(
            metadata={"filename": "note.pdf", "page": 3},
            get_content=lambda: "new content",
            node_id="node-1",
        )
        indexer.index_chunks([chunk], [[1.0, 2.0]])

        payload = index_client.upsert.call_args.kwargs
        assert payload["vectors"][0]["metadata"]["filename"] == "note.pdf"
        assert payload["vectors"][0]["metadata"]["page"] == 3
        assert payload["vectors"][0]["metadata"]["chunk_id"]
        index_client.delete.assert_not_called()
        index_client.list.assert_called_once_with(namespace="phase3-test", limit=100)


def test_indexer_generates_stable_ids_for_same_document_chunk_content():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client

        indexer = Indexer(
            api_key="test-key",
            index_name="college-notes",
            namespace="phase3-test",
        )

        chunk_a = SimpleNamespace(
            metadata={"document_id": "doc-abc", "filename": "note.pdf"},
            get_content=lambda: "same text",
            node_id="node-1",
        )
        chunk_b = SimpleNamespace(
            metadata={"document_id": "doc-abc", "filename": "note.pdf"},
            get_content=lambda: "same text",
            node_id="node-2",
        )

        assert indexer._build_record_id(chunk_a, 0) == indexer._build_record_id(chunk_b, 0)


def test_indexer_repeated_upload_has_stable_ids_and_same_filename_replaces_chunks():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        index_client.list.return_value = []
        indexer = Indexer("test-key", "college-notes", namespace="phase3-test")
        chunk = SimpleNamespace(
            metadata={"document_id": "doc-id", "filename": "notes.docx"},
            get_content=lambda: "same content",
        )

        indexer.index_chunks([chunk], [[1.0]])
        first_id = index_client.upsert.call_args.kwargs["vectors"][0]["id"]
        indexer.index_chunks([chunk], [[1.0]])
        second_id = index_client.upsert.call_args.kwargs["vectors"][0]["id"]

        assert first_id == second_id
        assert index_client.upsert.call_count == 2
        index_client.delete.assert_not_called()


def test_indexer_modified_same_filename_deletes_old_ids_only_after_upsert():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        indexer = Indexer("test-key", "college-notes", namespace="phase3-test")
        old_chunk = SimpleNamespace(
            metadata={"document_id": "doc-id", "filename": "notes.docx"},
            get_content=lambda: "old content",
        )
        old_id = indexer._build_record_id(old_chunk, 0)
        new_chunk = SimpleNamespace(
            metadata={"document_id": "doc-id", "filename": "notes.docx"},
            get_content=lambda: "updated content",
        )
        index_client.list.return_value = [
            SimpleNamespace(vectors=[SimpleNamespace(id=old_id)])
        ]
        index_client.fetch.return_value = SimpleNamespace(
            vectors={
                old_id: SimpleNamespace(metadata={"filename": "notes.docx"}),
                "other-id": SimpleNamespace(metadata={"filename": "other.docx"}),
            }
        )
        indexer.index_chunks([new_chunk], [[4.0]])
        new_id = index_client.upsert.call_args.kwargs["vectors"][0]["id"]

        assert new_id != old_id
        assert [call[0] for call in index_client.method_calls] == [
            "upsert", "list", "fetch", "delete"
        ]
        assert index_client.delete.call_args.kwargs["ids"] == [old_id]
        assert index_client.delete.call_args.kwargs["namespace"] == "phase3-test"


def test_indexer_reduced_chunk_count_deletes_only_stale_same_filename_chunks():
    index_client = Mock()
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        indexer = Indexer("test-key", "college-notes", namespace="phase3-test")
        old_ids = ["old-0", "old-1", "old-2"]
        index_client.list.return_value = [
            SimpleNamespace(
                vectors=[SimpleNamespace(id=vector_id) for vector_id in old_ids]
            )
        ]
        index_client.fetch.return_value = SimpleNamespace(
            vectors={
                vector_id: SimpleNamespace(metadata={"filename": "notes.pdf"})
                for vector_id in old_ids
            }
        )
        new_chunk = SimpleNamespace(
            metadata={"document_id": "doc-id", "filename": "notes.pdf"},
            get_content=lambda: "only remaining chunk",
        )

        indexer.index_chunks([new_chunk], [[1.0]])

        assert index_client.delete.call_args.kwargs["ids"] == old_ids
        assert index_client.delete.call_args.kwargs["namespace"] == "phase3-test"


def test_indexer_does_not_delete_existing_chunks_when_upsert_fails():
    index_client = Mock()
    index_client.upsert.side_effect = RuntimeError("upsert failed")
    with patch("app.ingestion.indexer.Pinecone") as mock_pinecone:
        mock_pinecone.return_value.Index.return_value = index_client
        indexer = Indexer("test-key", "college-notes", namespace="phase3-test")
        chunk = SimpleNamespace(
            metadata={"document_id": "doc-id", "filename": "notes.pdf"},
            get_content=lambda: "replacement content",
        )

        with pytest.raises(RuntimeError, match="upsert failed"):
            indexer.index_chunks([chunk], [[1.0]])

        index_client.delete.assert_not_called()


def test_document_loader_preserves_original_filename_and_pdf_page(monkeypatch, tmp_path):
    from llama_index.core import Document

    documents = [Document(text="page text", metadata={"page_label": "4"})]
    monkeypatch.setattr(
        "app.ingestion.loader.PDFReader",
        lambda: SimpleNamespace(load_data=lambda file: documents),
    )
    path = tmp_path / "temporary.pdf"
    path.write_bytes(b"pdf bytes")

    loaded = DocumentLoader().load(str(path), filename="original.pdf")

    assert loaded[0].metadata["filename"] == "original.pdf"
    assert loaded[0].metadata["page_number"] == 4
    assert loaded[0].metadata["document_id"]


def test_chunker_preserves_document_identity_filename_and_page_metadata():
    from llama_index.core import Document

    document = Document(
        text="A sentence about storage. " * 40,
        metadata={
            "document_id": "stable-document-id",
            "filename": "storage.pdf",
            "page_number": 2,
        },
    )

    chunks = DocumentChunker(chunk_size=64, chunk_overlap=0).chunk([document])

    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["document_id"] == "stable-document-id"
        assert chunk.metadata["filename"] == "storage.pdf"
        assert chunk.metadata["page_number"] == 2
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_id"]


def test_upload_keeps_old_file_if_ingestion_fails_and_atomically_replaces_on_success(
    monkeypatch,
    tmp_path,
):
    import asyncio

    from starlette.datastructures import UploadFile

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "data" / "documents" / "notes.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old document")

    failing_service = Mock()
    failing_service.ingest.side_effect = RuntimeError("indexing failed")
    failed_upload = UploadFile(filename="notes.pdf", file=BytesIO(b"new document"))
    with pytest.raises(RuntimeError, match="indexing failed"):
        asyncio.run(upload_document(failed_upload, failing_service))
    assert target.read_bytes() == b"old document"
    assert list(target.parent.glob(".ingesting-*")) == []

    successful_service = Mock()
    successful_service.ingest.return_value = {"filename": "notes.pdf", "chunks_indexed": 1}
    successful_upload = UploadFile(filename="notes.pdf", file=BytesIO(b"new document"))
    result = asyncio.run(upload_document(successful_upload, successful_service))

    assert result == {"filename": "notes.pdf", "chunks_indexed": 1}
    assert target.read_bytes() == b"new document"
    args, kwargs = successful_service.ingest.call_args
    assert kwargs["filename"] == "notes.pdf"
    assert Path(args[0]).name.startswith(".ingesting-")

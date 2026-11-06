from unittest.mock import Mock

import pytest
from starlette.requests import Request as StarletteRequest

from app.api.routes_chat import chat
from app.api.schemas import ChatMessage, ChatRequest
from app.core import dependencies
from app.retrieval.query_rewriter import QueryRewriter
from app.routing import QueryRouter
from app.services.rag_service import RAGService


def make_http_request(accept="text/plain"):
    return StarletteRequest({
        "type": "http",
        "headers": [(b"accept", accept.encode("ascii"))],
    })


@pytest.mark.parametrize(
    ("query", "history", "expected_route", "should_retrieve"),
    [
        ("hello there", None, "conversation", False),
        ("thanks", None, "conversation", False),
        ("rephrase that in simpler terms", [{"role": "assistant", "content": "The database is a structured system."}], "contextual", False),
        ("what is a database?", None, "retrieval", True),
        ("can you explain it more?", [{"role": "user", "content": "what is a database?"}], "retrieval", True),
        ("this one", [], "clarification", False),
        ("", None, "guardrail", False),
    ],
)
def test_query_router_routes_queries(query, history, expected_route, should_retrieve):
    decision = QueryRouter().classify(query, history)

    assert decision.route == expected_route
    assert decision.should_retrieve is should_retrieve


def test_greetings_bypass_extraction_and_retrieval_calls():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "not used"}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    rewriter.rewrite.return_value = "hello"
    llm = Mock()
    llm.generate_stream.return_value = iter(["greeting response"])

    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
        query_router=QueryRouter(),
    )

    answer = list(service.generate_answer("hello there"))

    assert answer == ["I can help with your notes and prior context. Ask a question about the uploaded material and I will use it to answer."]
    retriever.retrieve.assert_not_called()
    rewriter.rewrite.assert_not_called()
    llm.generate_stream.assert_not_called()


def test_contextual_rephrase_bypasses_embedding_and_pinecone_calls():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "not used"}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    rewriter.rewrite.return_value = "rephrase"
    llm = Mock()
    llm.generate_stream.return_value = iter(["shortened context"])

    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
        query_router=QueryRouter(),
    )

    history = [{"role": "assistant", "content": "The database stores records in a structured way."}]
    answer = list(service.generate_answer("rephrase that in simpler terms", history))

    assert answer == ["shortened context"]
    retriever.retrieve.assert_not_called()
    rewriter.rewrite.assert_not_called()
    llm.generate_stream.assert_called_once()

    prompt = llm.generate_stream.call_args[0][0]
    assert "Previous assistant answer" in prompt
    assert "The database stores records in a structured way." in prompt


def test_standalone_document_query_skips_rewriting():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "Database means a structured collection of records."}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    rewriter.rewrite.return_value = "database definition"
    llm = Mock()
    llm.generate_stream.return_value = iter(["A database stores structured records."])

    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
        query_router=QueryRouter(),
    )

    answer = list(service.generate_answer("what is a database?"))

    assert answer == ["A database stores structured records."]
    rewriter.rewrite.assert_not_called()
    retriever.retrieve.assert_called_once_with("what is a database?")
    llm.generate_stream.assert_called_once()


def test_clarification_and_guardrail_bypass_retrieval():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "not used"}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    rewriter.rewrite.return_value = "unclear"
    llm = Mock()
    llm.generate_stream.return_value = iter(["ignored"])

    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
        query_router=QueryRouter(),
    )

    assert list(service.generate_answer("this one")) == ["I need a bit more context before I can answer accurately. Please specify what you want to know."]
    assert list(service.generate_answer("")) == ["Please provide a clear question or request so I can help with the uploaded document context."]
    retriever.retrieve.assert_not_called()
    rewriter.rewrite.assert_not_called()
    llm.generate_stream.assert_not_called()


def test_follow_up_without_sufficient_context_routes_to_retrieval_or_clarification():
    query_router = QueryRouter()

    assert query_router.classify("what about it", []).route == "retrieval"
    assert query_router.classify("this one", []).route == "clarification"


@pytest.mark.parametrize(
    "query",
    [
        "Explain the three storage modes mentioned in simpler words",
        "Make the previous answer easier to understand",
        "Explain those concepts more simply",
        "Explain storage modes in simpler words",
    ],
)
def test_contextual_rephrasing_variants_require_and_match_prior_answer(query):
    history = [
        {
            "role": "assistant",
            "content": "The three storage modes are block, file, and object storage.",
        }
    ]

    decision = QueryRouter().classify(query, history)

    assert decision.route == "contextual"
    assert decision.should_retrieve is False


def test_contextual_rephrasing_does_not_use_unrelated_history():
    history = [{"role": "assistant", "content": "A database stores structured records."}]

    decision = QueryRouter().classify(
        "Explain storage modes in simpler words",
        history,
    )

    assert decision.route == "retrieval"
    assert decision.should_retrieve is True


def test_factual_pronoun_follow_up_still_uses_retrieval():
    history = [
        {"role": "assistant", "content": "The service uses a client-server architecture."}
    ]

    decision = QueryRouter().classify("What is its default port?", history)

    assert decision.route == "retrieval"
    assert decision.should_retrieve is True


@pytest.mark.parametrize(
    "query",
    [
        "Explain the three storage modes mentioned in simpler words",
        "Make the previous answer easier to understand",
        "Explain those concepts more simply",
    ],
)
def test_contextual_variants_bypass_query_rewriting_and_retrieval(query):
    retriever = Mock()
    rewriter = Mock()
    llm = Mock()
    llm.generate_stream.return_value = iter(["simplified answer"])
    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
    )
    history = [
        {
            "role": "assistant",
            "content": "The three storage modes are block, file, and object storage.",
        }
    ]

    assert list(service.generate_answer(query, history)) == ["simplified answer"]
    rewriter.rewrite.assert_not_called()
    retriever.retrieve.assert_not_called()
    llm.generate_stream.assert_called_once()


def test_factual_pronoun_follow_up_invokes_query_rewriting_and_retrieval():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "The default port is 5432."}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = True
    rewriter.rewrite.return_value = "default port"
    llm = Mock()
    llm.generate_stream.return_value = iter(["The default port is 5432."])
    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
    )
    history = [
        {"role": "assistant", "content": "The service uses a client-server architecture."}
    ]

    assert list(service.generate_answer("What is its default port?", history)) == [
        "The default port is 5432."
    ]
    rewriter.rewrite.assert_called_once()
    retriever.retrieve.assert_called_once_with("default port")


def test_chat_endpoint_normalizes_schema_history_without_changing_order():
    rag_service = Mock()
    request = ChatRequest(
        query="What is its default port?",
        history=[
            ChatMessage(role="user", content="Tell me about Orion Database."),
            ChatMessage(role="assistant", content="It is a relational database."),
        ],
    )

    response = chat(
        request,
        http_request=make_http_request(),
        rag_service=rag_service,
    )

    assert response.media_type == "text/plain"
    rag_service.generate_answer.assert_called_once_with(
        "What is its default port?",
        [
            {"role": "user", "content": "Tell me about Orion Database."},
            {"role": "assistant", "content": "It is a relational database."},
        ],
    )


def test_three_turn_chat_routes_contextual_rephrase_without_retrieval():
    retriever = Mock()
    retriever.retrieve.side_effect = [
        [{"text": "Orion Database was developed by the Orion team."}],
        [{"text": "It supports block, file, and object storage."}],
    ]
    rewriter = QueryRewriter(
        base_url="http://localhost:11434",
        model="qwen3:8b",
        context_window=4096,
    )
    rewriter.client = Mock()
    rewriter.client.generate.return_value = {"response": "Orion storage modes"}
    llm = Mock()
    llm.generate_stream.side_effect = [
        iter(["The Orion team developed the Orion Database."]),
        iter(["It supports block, file, and object storage."]),
        iter(["Block stores data in chunks, file uses folders, and object stores complete items."]),
    ]
    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
    )

    def post_chat(query, history):
        response = chat(
            ChatRequest(query=query, history=history),
            http_request=make_http_request(),
            rag_service=service,
        )

        async def collect():
            return "".join(
                [chunk async for chunk in response.body_iterator]
            )

        import asyncio

        return asyncio.run(collect())

    first_answer = post_chat("Who developed the Orion Database?", [])
    history = [
        ChatMessage(role="user", content="Who developed the Orion Database?"),
        ChatMessage(role="assistant", content=first_answer),
    ]
    second_answer = post_chat("What storage modes does it support?", history)
    history.extend(
        [
            ChatMessage(role="user", content="What storage modes does it support?"),
            ChatMessage(role="assistant", content=second_answer),
        ]
    )
    third_answer = post_chat(
        "Explain the three storage modes mentioned in simpler words.",
        history,
    )

    assert first_answer == "The Orion team developed the Orion Database."
    assert second_answer == "It supports block, file, and object storage."
    assert third_answer.startswith("Block stores data")
    assert retriever.retrieve.call_count == 2
    assert rewriter.client.generate.call_count == 1
    assert llm.generate_stream.call_count == 3


def test_standalone_question_skips_rewriting_even_with_existing_history():
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "Orion Database was developed in 2020."}]
    rewriter = QueryRewriter(
        base_url="http://localhost:11434",
        model="qwen3:8b",
        context_window=4096,
    )
    rewriter.client = Mock()
    llm = Mock()
    llm.generate_stream.return_value = iter(["The Orion Database was developed in 2020."])
    service = RAGService(
        retriever=retriever,
        ollama_client=llm,
        query_rewriter=rewriter,
    )

    answer = list(
        service.generate_answer(
            "Who developed the Orion Database?",
            [{"role": "assistant", "content": "We were discussing another topic."}],
        )
    )

    assert answer == ["The Orion Database was developed in 2020."]
    rewriter.client.generate.assert_not_called()
    retriever.retrieve.assert_called_once_with("Who developed the Orion Database?")


def test_retriever_dependency_reuses_initialized_instance(monkeypatch):
    dependencies.get_retriever.cache_clear()
    retriever = Mock()
    constructor = Mock(return_value=retriever)
    monkeypatch.setattr(dependencies, "Retriever", constructor)

    try:
        first = dependencies.get_retriever()
        second = dependencies.get_retriever()
    finally:
        dependencies.get_retriever.cache_clear()

    assert first is second is retriever
    constructor.assert_called_once()


def test_query_rewriter_only_rewrites_context_references_and_disables_thinking():
    rewriter = QueryRewriter(
        base_url="http://localhost:11434",
        model="qwen3:8b",
        context_window=4096,
    )
    rewriter.client = Mock()
    rewriter.client.generate.return_value = {"response": "Orion default port"}
    history = [{"role": "assistant", "content": "The Orion Database runs as a service."}]

    assert rewriter.needs_rewrite("Who developed the Orion Database?", history) is False
    assert rewriter.needs_rewrite("What is its default port?", history) is True
    assert rewriter.rewrite("What is its default port?", history) == "Orion default port"

    kwargs = rewriter.client.generate.call_args.kwargs
    assert kwargs["think"] is False
    assert kwargs["options"] == {"num_ctx": 4096}


def test_optional_rewriter_model_falls_back_to_generation_model(monkeypatch):
    monkeypatch.setattr(dependencies.settings, "OLLAMA_REWRITE_MODEL", None)
    with pytest.MonkeyPatch.context() as patch:
        client_constructor = Mock()
        patch.setattr("app.retrieval.query_rewriter.Client", client_constructor)
        default_rewriter = dependencies.get_query_rewriter()
        assert default_rewriter.model == dependencies.settings.OLLAMA_GENERATION_MODEL

        patch.setattr(dependencies.settings, "OLLAMA_REWRITE_MODEL", "qwen3:1.7b")
        smaller_rewriter = dependencies.get_query_rewriter()
        assert smaller_rewriter.model == "qwen3:1.7b"
        client_constructor.assert_called()


def test_empty_retrieval_abstains_without_generating():
    retriever = Mock()
    retriever.retrieve.return_value = []
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    llm = Mock()
    service = RAGService(retriever, llm, rewriter)

    assert list(service.generate_answer("what is an unrelated device?")) == [
        "I could not find sufficiently relevant context in the uploaded notes for that question."
    ]
    llm.generate_stream.assert_not_called()


def test_service_logs_rewrite_first_token_and_total_timings(caplog):
    retriever = Mock()
    retriever.retrieve.return_value = [{"text": "A database stores records."}]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    llm = Mock()
    llm.generate_stream.return_value = iter(["Answer"])
    service = RAGService(retriever, llm, rewriter)

    with caplog.at_level("INFO"):
        assert list(service.generate_answer("what is a database?")) == ["Answer"]

    messages = [record.getMessage() for record in caplog.records]
    assert any("query_rewrite_ms=" in message for message in messages)
    assert any("time_to_first_token_ms=" in message for message in messages)
    assert any("total_response_ms=" in message for message in messages)


def test_high_similarity_is_not_presented_as_proof_and_partial_context_is_marked():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"text": "The notes describe block storage.", "filename": "storage.pdf", "_score": 0.94}
    ]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    llm = Mock()
    llm.generate_stream.return_value = iter(["The notes describe block storage only."])
    service = RAGService(retriever, llm, rewriter)

    assert list(service.generate_answer("compare all three storage modes")) == [
        "The notes describe block storage only."
    ]
    prompt = llm.generate_stream.call_args.args[0]
    assert "similarity scores only rank passages" in prompt
    assert "supports only part of the question" in prompt
    assert "0.94" not in prompt


def test_sources_are_emitted_without_inventing_missing_metadata():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "text": "A passage",
            "filename": "notes.pdf",
            "page_number": 4,
            "chunk_id": "stable-chunk-id",
            "_score": 0.9,
        },
        {"text": "Metadata-free passage", "_score": 0.8},
    ]
    rewriter = Mock()
    rewriter.needs_rewrite.return_value = False
    llm = Mock()
    llm.generate_stream.return_value = iter(["Answer"])
    service = RAGService(retriever, llm, rewriter)

    events = list(service.generate_events("what is described?", []))

    assert events[0] == {"event": "token", "data": "Answer"}
    assert events[-1] == {
        "event": "sources",
        "data": [
            {"filename": "notes.pdf", "page_number": 4}
        ],
    }


def test_chat_sse_exposes_source_event_and_plain_text_api_remains_compatible():
    rag_service = Mock()
    rag_service.generate_events.return_value = iter(
        [
            {"event": "token", "data": "Answer"},
            {"event": "sources", "data": [{"filename": "notes.pdf"}]},
        ]
    )
    rag_service.generate_answer.return_value = iter(["Answer"])
    request = ChatRequest(query="what is this?", history=[])

    sse_response = chat(
        request,
        http_request=make_http_request("text/event-stream"),
        rag_service=rag_service,
    )

    async def read_sse():
        return "".join([chunk async for chunk in sse_response.body_iterator])

    import asyncio

    sse_body = asyncio.run(read_sse())
    assert sse_response.media_type == "text/event-stream"
    assert "event: token" in sse_body
    assert "event: sources" in sse_body

    text_response = chat(
        request,
        http_request=make_http_request(),
        rag_service=rag_service,
    )
    assert text_response.media_type == "text/plain"

    async def read_text():
        return "".join([chunk async for chunk in text_response.body_iterator])

    assert asyncio.run(read_text()) == "Answer"

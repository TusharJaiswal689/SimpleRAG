from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


RouteName = Literal["conversation", "contextual", "retrieval", "clarification", "guardrail"]


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    should_retrieve: bool
    rationale: str


class QueryRouter:
    """Classify user input before any document retrieval or model generation."""

    GREETINGS = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "afternoon",
        "evening",
    }

    ACKNOWLEDGEMENTS = {
        "thanks",
        "thank you",
        "okay",
        "ok",
        "sure",
        "got it",
        "understood",
    }

    CONTEXTUAL_PATTERNS = (
        "rephrase",
        "reword",
        "summarize",
        "simplify",
        "simpler",
        "more simply",
        "easier to understand",
        "break down",
        "put simply",
        "restate",
        "explain in simpler terms",
        "explain it simply",
        "explain more simply",
        "translate it into simple terms",
        "what did i just say",
        "what did i say",
        "what was my previous answer",
        "clarify the previous answer",
        "tell me in simpler words",
    )

    CONTEXT_REFERENCES = (
        "previous answer",
        "previous response",
        "prior answer",
        "prior response",
        "last answer",
        "last response",
        "above",
        "earlier",
        "mentioned",
        "as you said",
        "you said",
        "that",
        "it",
        "those",
        "these",
        "them",
        "this",
    )

    FOLLOW_UP_PATTERNS = (
        "what about it",
        "what about this",
        "what about that",
        "tell me more",
        "explain it more",
        "can you explain it",
        "what does it mean",
        "can you elaborate",
        "how does it work",
    )

    RETRIEVAL_MARKERS = (
        "what",
        "why",
        "how",
        "explain",
        "summarize",
        "compare",
        "list",
        "which",
        "who",
        "where",
        "when",
        "describe",
        "define",
        "find",
        "analyze",
        "contrast",
        "detail",
        "based on the notes",
        "in the uploaded file",
    )

    CONTEXT_OVERLAP_STOP_WORDS = {
        "a", "an", "and", "answer", "break", "can", "could", "do", "down",
        "easier", "explain", "in", "it", "make", "me", "more", "of", "please",
        "rephrase", "restate", "reword", "simpler", "simplify", "simply",
        "summarize", "tell", "terms", "that", "the", "this", "to", "understand",
        "use", "words", "you",
    }

    def classify(self, query: str, history: list[dict] | None = None) -> RouteDecision:
        cleaned = (query or "").strip()
        history = history or []

        if not cleaned:
            return RouteDecision(
                route="guardrail",
                should_retrieve=False,
                rationale="Empty query was rejected before retrieval.",
            )

        normalized = cleaned.lower()

        if normalized in self.GREETINGS or normalized in self.ACKNOWLEDGEMENTS:
            return RouteDecision(
                route="conversation",
                should_retrieve=False,
                rationale="The message is social or conversational and does not require document retrieval.",
            )

        if normalized in {"bye", "goodbye", "see you later"}:
            return RouteDecision(
                route="conversation",
                should_retrieve=False,
                rationale="The user is closing the conversation rather than asking about the documents.",
            )

        if self._is_contextual_rephrase(normalized, history):
            return RouteDecision(
                route="contextual",
                should_retrieve=False,
                rationale="The request is to rephrase or simplify existing conversation context rather than query the document store.",
            )

        if self._is_ambiguous_reference(normalized, history):
            return RouteDecision(
                route="clarification",
                should_retrieve=False,
                rationale="The request references prior context without enough information to determine whether retrieval is needed.",
            )

        if self._looks_like_follow_up(normalized, history):
            return RouteDecision(
                route="retrieval",
                should_retrieve=True,
                rationale="The message depends on earlier context but also requires evidence from the document store.",
            )

        if self._requires_document_lookup(normalized):
            return RouteDecision(
                route="retrieval",
                should_retrieve=True,
                rationale="The request asks for subject-specific content that should be retrieved from indexed material.",
            )

        return RouteDecision(
            route="conversation",
            should_retrieve=False,
            rationale="The message is generic conversation without a retrieval need.",
        )

    def _is_contextual_rephrase(self, query: str, history: list[dict]) -> bool:
        assistant_content = self._recent_assistant_content(history)
        if not assistant_content:
            return False

        query_tokens = self._tokens(query)
        has_rephrasing_intent = any(
            self._contains_phrase(query_tokens, pattern)
            for pattern in self.CONTEXTUAL_PATTERNS
        )
        if not has_rephrasing_intent:
            return False

        if any(
            self._contains_phrase(query_tokens, reference)
            for reference in self.CONTEXT_REFERENCES
        ):
            return True

        query_content = {
            token
            for token in query_tokens
            if len(token) > 2 and token not in self.CONTEXT_OVERLAP_STOP_WORDS
        }
        prior_tokens = {
            token
            for content in assistant_content
            for token in self._tokens(content)
        }
        return bool(query_content & prior_tokens)

    def _is_ambiguous_reference(self, query: str, history: list[dict]) -> bool:
        mention_tokens = (
            "it", "this", "that", "they", "those", "the previous",
            "the former", "the latter", "this one", "that one",
        )
        query_tokens = self._tokens(query)
        if len(query_tokens) <= 6 and any(
            self._contains_phrase(query_tokens, token) for token in mention_tokens
        ):
            if not self._requires_document_lookup(query):
                return True
        if not history and query_tokens in {
            ("this", "one"), ("that", "one"), ("this",), ("that",),
        }:
            return True
        return False

    def _looks_like_follow_up(self, query: str, history: list[dict]) -> bool:
        if not self._recent_assistant_content(history):
            return False

        query_tokens = self._tokens(query)
        if any(
            self._contains_phrase(query_tokens, pattern)
            for pattern in self.FOLLOW_UP_PATTERNS
        ):
            return True
        if len(query_tokens) <= 8 and any(
            msg.get("role") == "user" for msg in history[-3:]
        ):
            if any(
                self._contains_phrase(query_tokens, token)
                for token in ("it", "this", "that", "they", "those")
            ):
                if self._requires_document_lookup(query):
                    return True
        return False

    def _requires_document_lookup(self, query: str) -> bool:
        query_tokens = self._tokens(query)
        return any(
            self._contains_phrase(query_tokens, marker)
            for marker in self.RETRIEVAL_MARKERS
        )

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", text.casefold()))

    @staticmethod
    def _contains_phrase(tokens: tuple[str, ...], phrase: str) -> bool:
        phrase_tokens = QueryRouter._tokens(phrase)
        phrase_length = len(phrase_tokens)
        return any(
            tokens[index:index + phrase_length] == phrase_tokens
            for index in range(len(tokens) - phrase_length + 1)
        )

    @staticmethod
    def _recent_assistant_content(history: list[dict]) -> list[str]:
        return [
            str(message.get("content", "")).strip()
            for message in history[-5:]
            if isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "assistant"
            and str(message.get("content", "")).strip()
        ]

"""Document analysis: payload normalisation and large-document handling."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pytest

from app.ai.analyzer import MAX_SECTIONS, DocumentAnalysis, DocumentAnalyzer
from app.ai.base import ChatMessage, LLMProvider, StructuredResult, TokenUsage
from app.ai.schemas import DOCUMENT_ANALYSIS_SCHEMA, SECTION_SUMMARY_SCHEMA
from app.core.config import IngestionSettings
from app.core.constants import DocumentCategory, Sentiment

VALID_PAYLOAD: dict[str, Any] = {
    "summary": "A remote work policy covering eligibility and security.",
    "key_points": ["Ninety days of service required.", "VPN is mandatory."],
    "tags": ["Remote-Work", "SECURITY"],
    "category": "hr",
    "sentiment": "neutral",
    "language": "EN",
    "entities": [{"name": "ACME", "type": "organization"}],
}


class ScriptedLLM(LLMProvider):
    """LLM double returning queued payloads and recording its calls."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    @property
    def model(self) -> str:
        return "test-model"

    def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        self.calls.append((schema_name, list(messages)))
        payload = self._payloads.pop(0) if self._payloads else VALID_PAYLOAD
        return StructuredResult(
            data=payload,
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=50, cost_usd=Decimal("0.001")
            ),
            model=self.model,
        )

    def complete(self, messages, *, max_tokens=None, temperature=None):
        raise NotImplementedError

    def stream(self, messages, *, max_tokens=None, temperature=None):
        raise NotImplementedError


@pytest.fixture
def settings() -> IngestionSettings:
    return IngestionSettings(max_analysis_chars=1000)


class TestNormalisation:
    def test_valid_payload_is_mapped_to_domain_types(self) -> None:
        analysis = DocumentAnalysis.from_payload(
            VALID_PAYLOAD, model="m", usage=TokenUsage(), was_truncated=False
        )

        assert analysis.category is DocumentCategory.HR
        assert analysis.sentiment is Sentiment.NEUTRAL
        assert analysis.language == "en"

    def test_tags_are_lowercased(self) -> None:
        analysis = DocumentAnalysis.from_payload(
            VALID_PAYLOAD, model="m", usage=TokenUsage(), was_truncated=False
        )
        assert analysis.tags == ["remote-work", "security"]

    def test_unknown_category_falls_back_instead_of_failing(self) -> None:
        payload = VALID_PAYLOAD | {"category": "interplanetary"}
        analysis = DocumentAnalysis.from_payload(
            payload, model="m", usage=TokenUsage(), was_truncated=False
        )
        assert analysis.category is DocumentCategory.OTHER

    def test_unknown_sentiment_falls_back(self) -> None:
        payload = VALID_PAYLOAD | {"sentiment": "ecstatic"}
        analysis = DocumentAnalysis.from_payload(
            payload, model="m", usage=TokenUsage(), was_truncated=False
        )
        assert analysis.sentiment is Sentiment.NEUTRAL

    def test_blank_list_entries_are_dropped(self) -> None:
        payload = VALID_PAYLOAD | {"key_points": ["Real point.", "   ", ""]}
        analysis = DocumentAnalysis.from_payload(
            payload, model="m", usage=TokenUsage(), was_truncated=False
        )
        assert analysis.key_points == ["Real point."]

    def test_malformed_entities_are_discarded(self) -> None:
        payload = VALID_PAYLOAD | {
            "entities": [{"name": "ACME", "type": "organization"}, "junk", {}]
        }
        analysis = DocumentAnalysis.from_payload(
            payload, model="m", usage=TokenUsage(), was_truncated=False
        )
        assert analysis.entities == [{"name": "ACME", "type": "organization"}]

    def test_missing_optional_fields_use_defaults(self) -> None:
        analysis = DocumentAnalysis.from_payload(
            {"summary": "Just a summary."},
            model="m",
            usage=TokenUsage(),
            was_truncated=False,
        )
        assert analysis.key_points == []
        assert analysis.tags == []
        assert analysis.language == "en"


class TestSinglePassAnalysis:
    def test_small_documents_use_one_call(self, settings: IngestionSettings) -> None:
        llm = ScriptedLLM([VALID_PAYLOAD])
        analysis = DocumentAnalyzer(llm, settings).analyze(
            "Short policy text.", filename="policy.txt"
        )

        assert len(llm.calls) == 1
        assert llm.calls[0][0] == "document_analysis"
        assert analysis.was_truncated is False

    def test_filename_is_included_in_the_prompt(
        self, settings: IngestionSettings
    ) -> None:
        llm = ScriptedLLM([VALID_PAYLOAD])
        DocumentAnalyzer(llm, settings).analyze("Text.", filename="contract.pdf")

        assert "contract.pdf" in llm.calls[0][1][-1].content

    def test_empty_text_is_rejected_before_any_call(
        self, settings: IngestionSettings
    ) -> None:
        llm = ScriptedLLM([])
        with pytest.raises(ValueError, match="empty document"):
            DocumentAnalyzer(llm, settings).analyze("   ", filename="a.txt")
        assert llm.calls == []


class TestLargeDocumentAnalysis:
    @staticmethod
    def _large_text(paragraphs: int = 60) -> str:
        return "\n\n".join(["Policy paragraph with content. " * 40] * paragraphs)

    def test_large_documents_are_condensed_then_analysed(
        self, settings: IngestionSettings
    ) -> None:
        section = {"summary": "Section summary.", "key_points": ["A fact."]}
        llm = ScriptedLLM([section] * 20 + [VALID_PAYLOAD])

        DocumentAnalyzer(llm, settings).analyze(self._large_text(), filename="big.txt")

        schema_names = [name for name, _ in llm.calls]
        assert schema_names.count("section_summary") >= 1
        assert schema_names[-1] == "document_analysis"

    def test_usage_is_summed_across_every_call(
        self, settings: IngestionSettings
    ) -> None:
        section = {"summary": "S.", "key_points": []}
        llm = ScriptedLLM([section] * 20 + [VALID_PAYLOAD])

        analysis = DocumentAnalyzer(llm, settings).analyze(
            self._large_text(), filename="big.txt"
        )

        assert analysis.usage.cost_usd == Decimal("0.001") * len(llm.calls)

    def test_section_count_is_capped(self, settings: IngestionSettings) -> None:
        section = {"summary": "S.", "key_points": []}
        llm = ScriptedLLM([section] * 100 + [VALID_PAYLOAD])

        DocumentAnalyzer(llm, settings).analyze(
            self._large_text(paragraphs=4000), filename="huge.txt"
        )

        sections = sum(1 for name, _ in llm.calls if name == "section_summary")
        assert sections <= MAX_SECTIONS


class TestSchemas:
    @pytest.mark.parametrize(
        "schema", [DOCUMENT_ANALYSIS_SCHEMA, SECTION_SUMMARY_SCHEMA]
    )
    def test_strict_mode_requirements_are_met(self, schema: dict[str, Any]) -> None:
        """OpenAI strict mode requires every property to appear in required."""
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    def test_category_enum_matches_the_domain(self) -> None:
        values = DOCUMENT_ANALYSIS_SCHEMA["properties"]["category"]["enum"]
        assert set(values) == {member.value for member in DocumentCategory}

    def test_nested_entity_schema_is_also_strict(self) -> None:
        entity = DOCUMENT_ANALYSIS_SCHEMA["properties"]["entities"]["items"]
        assert entity["additionalProperties"] is False
        assert set(entity["required"]) == set(entity["properties"])

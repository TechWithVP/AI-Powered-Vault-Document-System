"""JSON schemas for structured AI output.

Each schema is enforced by the provider rather than parsed defensively, so the
model cannot return a shape the application does not expect. OpenAI strict mode
requires every property to be listed in "required" and additionalProperties to
be false, so optionality is expressed with nullable types instead.
"""

from __future__ import annotations

from typing import Any

from app.core.constants import DocumentCategory, Sentiment

#: Bounds chosen so a summary is useful at a glance without becoming a second
#: document, and so tag lists stay filterable rather than sprawling.
MAX_KEY_POINTS = 6
MAX_TAGS = 7
MAX_ENTITIES = 12

ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "date",
    "money",
    "product",
    "other",
]

DOCUMENT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "key_points",
        "tags",
        "category",
        "sentiment",
        "language",
        "entities",
    ],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Three to five sentences describing what this document is and "
                "what it contains. Written for someone who has not read it."
            ),
        },
        "key_points": {
            "type": "array",
            "maxItems": MAX_KEY_POINTS,
            "items": {"type": "string"},
            "description": (
                "The most consequential facts, obligations or findings, each a "
                "self-contained sentence."
            ),
        },
        "tags": {
            "type": "array",
            "maxItems": MAX_TAGS,
            "items": {"type": "string"},
            "description": (
                "Lowercase topical keywords for filtering, such as "
                "'remote-work' or 'data-retention'."
            ),
        },
        "category": {
            "type": "string",
            "enum": [member.value for member in DocumentCategory],
            "description": "The single best-fitting document category.",
        },
        "sentiment": {
            "type": "string",
            "enum": [member.value for member in Sentiment],
            "description": (
                "Overall tone of the document itself, not your opinion of it."
            ),
        },
        "language": {
            "type": "string",
            "description": "ISO 639-1 code of the dominant language, e.g. 'en'.",
        },
        "entities": {
            "type": "array",
            "maxItems": MAX_ENTITIES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                },
            },
            "description": "Named entities that matter for finding this document.",
        },
    },
}

#: Used when a document is too large to analyse in one pass: each section is
#: condensed first, then the condensations are analysed together.
SECTION_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "key_points"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "Two to three sentences covering this section only.",
        },
        "key_points": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
            "description": "Concrete facts from this section, not generalities.",
        },
    },
}


#: Follow-up suggestions are returned as structured output so the UI can render
#: them as buttons without parsing a free-text list.
FOLLOW_UP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
            "description": (
                "Questions the user could ask next, answerable from the same "
                "excerpts."
            ),
        }
    },
}

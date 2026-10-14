"""Question classification.

Not every question sent to a document chat is a question *about the documents*.
Routing them all through retrieval produces two bad outcomes: questions about
the conversation itself get refused even though the answer is in the history,
and counting questions get answered confidently from a partial sample of the
document. Classifying first lets each kind be handled honestly.
"""

from __future__ import annotations

import re
from enum import StrEnum


class QuestionKind(StrEnum):
    """How a question should be answered."""

    #: Ordinary question about document content.
    DOCUMENT = "document"
    #: Question about this conversation, answerable from the transcript.
    CONVERSATION = "conversation"
    #: Counting, listing or totalling, where a partial sample is misleading.
    AGGREGATE = "aggregate"


def _pattern(*fragments: str) -> re.Pattern[str]:
    return re.compile("|".join(fragments), re.IGNORECASE)


#: Phrases that unambiguously refer to the exchange rather than the documents.
#: Deliberately narrow: "where is X mentioned" is about the document, so only
#: explicit references to the assistant, the user's own turns, or the numbered
#: excerpts count.
CONVERSATION_PATTERNS = _pattern(
    r"\byou (said|told|mentioned|wrote|answered|replied|shared|gave)\b",
    r"\byour (earlier|previous|last|first|above|prior) (answer|response|reply|message)\b",
    r"\b(earlier|previous|last|above|prior) (answer|response|reply|message)\b",
    r"\b(i|we) (just )?asked\b",
    r"\bdid (i|we) ask\b",
    r"\bmy (first|second|third|last|previous|earlier|\d+(st|nd|rd|th)) question\b",
    r"\b(first|second|third|\d+(st|nd|rd|th)) (question|excerpt)\b",
    r"\bthis (conversation|chat|convo|thread|session)\b",
    r"\bour (conversation|chat|discussion)\b",
    r"\bwhere did you get\b",
    r"\bwhich excerpt\b",
    r"\bso far\b.*\b(asked|discussed|covered)\b",
    r"\b(repeat|rephrase|summari[sz]e) (that|your|the above|our)\b",
)

#: Phrases implying the answer must cover the whole document, not a sample.
AGGREGATE_PATTERNS = _pattern(
    r"\bhow many\b",
    r"\bhow much (in )?total\b",
    r"\b(total|overall) number\b",
    r"\bcount (the|all|how)\b",
    r"\blist (all|every|out all)\b",
    r"\ball (the )?(of )?(exclusions?|conditions?|clauses?|sections?|items?|points?|benefits?|definitions?|charges?|fees?)\b",
    r"\bevery (exclusion|condition|clause|section|item|benefit|definition)\b",
    r"\b(complete|full|exhaustive) list\b",
    r"\benumerate\b",
)


class QuestionClassifier:
    """Decides how a question should be routed."""

    def __init__(
        self,
        conversation: re.Pattern[str] = CONVERSATION_PATTERNS,
        aggregate: re.Pattern[str] = AGGREGATE_PATTERNS,
    ) -> None:
        self._conversation = conversation
        self._aggregate = aggregate

    def classify(self, question: str, *, has_history: bool = False) -> QuestionKind:
        """Classify a question.

        Conversation questions are only meaningful once there is a transcript to
        answer from, so without history they fall through to the document path
        and are handled by the usual relevance gate.
        """
        text = question.strip()
        if not text:
            return QuestionKind.DOCUMENT

        if has_history and self._conversation.search(text):
            return QuestionKind.CONVERSATION
        if self._aggregate.search(text):
            return QuestionKind.AGGREGATE
        return QuestionKind.DOCUMENT

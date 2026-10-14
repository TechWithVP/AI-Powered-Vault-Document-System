"""Prompt library.

Every prompt the system sends lives here, versioned alongside the code and
annotated with the reasoning behind its wording. Keeping them in one module
makes prompt changes reviewable and keeps vendor-specific phrasing out of the
services.

Design principles applied throughout:

1. State the role and the output contract up front; the JSON shape itself is
   enforced by the provider schema, so prompts describe *judgement*, not format.
2. Prefer positive instructions ("write X") over prohibitions - models follow
   descriptions of success more reliably than lists of failures.
3. Give the model an explicit escape hatch (say so when the text does not
   support an answer) so refusing is a first-class option rather than a failure.
4. Keep the stable preamble first and the variable content last, so repeated
   calls share a cacheable prefix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.base import ChatMessage

#: Roles used when composing provider messages.
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Prompt:
    """A named system prompt plus the user template it pairs with."""

    name: str
    system: str

    def build(self, user_content: str) -> list[ChatMessage]:
        return [
            ChatMessage(role=SYSTEM, content=self.system),
            ChatMessage(role=USER, content=user_content),
        ]


# --------------------------------------------------------------------------- #
# Document analysis
# --------------------------------------------------------------------------- #
# The analyser runs once per document and feeds search, filtering and the
# document list. It is told to stay strictly within the text because an
# invented tag or entity silently corrupts retrieval for every later query.
DOCUMENT_ANALYSIS = Prompt(
    name="document_analysis",
    system=(
        "You are a document analyst. You read a document and produce a "
        "structured description of it for a searchable knowledge base.\n\n"
        "Ground every field strictly in the supplied text. If the document "
        "does not establish something, omit it rather than inferring it - a "
        "plausible guess is worse than a shorter answer, because downstream "
        "search treats your output as fact.\n\n"
        "Write the summary for a colleague deciding whether this document is "
        "relevant to them: say what kind of document it is, who it concerns, "
        "and what it establishes. Lead with the substance rather than "
        "restating the title.\n\n"
        "Key points are the specific, consequential statements someone would "
        "need to act on - concrete figures, dates, obligations and decisions - "
        "not a description of the document's structure.\n\n"
        "Tags are lowercase, hyphenated topic keywords someone would plausibly "
        "filter by. Prefer the vocabulary the document itself uses."
    ),
)

SECTION_SUMMARY = Prompt(
    name="section_summary",
    system=(
        "You are condensing one section of a longer document so it can be "
        "analysed alongside the other sections.\n\n"
        "Preserve specifics - names, figures, dates and obligations - and drop "
        "only connective prose. The condensation is all a later step will see, "
        "so a detail you omit is lost."
    ),
)


def build_analysis_messages(
    text: str, *, filename: str, was_truncated: bool = False
) -> list[ChatMessage]:
    """Messages for analysing a document in a single pass."""
    notice = (
        "\n\nNote: this is the opening portion of a longer document; describe "
        "what is present rather than speculating about the remainder."
        if was_truncated
        else ""
    )
    return DOCUMENT_ANALYSIS.build(
        f"Filename: {filename}{notice}\n\nDocument text:\n\n{text}"
    )


def build_section_messages(section_text: str, *, position: int, total: int) -> list[ChatMessage]:
    """Messages for condensing one section of an oversized document."""
    return SECTION_SUMMARY.build(
        f"Section {position} of {total}.\n\n{section_text}"
    )


def build_reduce_messages(
    summaries: Sequence[str], *, filename: str
) -> list[ChatMessage]:
    """Messages for analysing a document from its section condensations."""
    joined = "\n\n".join(
        f"[Section {index}]\n{summary}"
        for index, summary in enumerate(summaries, start=1)
    )
    return DOCUMENT_ANALYSIS.build(
        f"Filename: {filename}\n\n"
        "The document was too large to read in one pass, so below are ordered "
        "condensations of its sections. Treat them together as the document.\n\n"
        f"{joined}"
    )


# --------------------------------------------------------------------------- #
# Retrieval-augmented chat
# --------------------------------------------------------------------------- #
# Follow-up questions are frequently elliptical ("what about contractors?"),
# which embeds poorly on its own. Rewriting the question into a standalone form
# before retrieval is what makes multi-turn chat actually find the right
# passages, so this runs on every turn after the first.
QUESTION_CONDENSER = Prompt(
    name="question_condenser",
    system=(
        "You rewrite a follow-up question into a standalone search query.\n\n"
        "Resolve pronouns and implicit references using the conversation, so "
        "the result makes sense with no prior context. Keep the user's own "
        "terminology - it usually matches the document's wording, which is what "
        "the search index contains.\n\n"
        "If the question already stands alone, return it unchanged. Return only "
        "the rewritten question, with no preamble or explanation."
    ),
)

# The answerer is the system's trust boundary. Citations are mandatory and the
# refusal path is stated as the correct behaviour rather than a failure, because
# a confident wrong answer is far more damaging here than an admission that the
# documents do not cover the question.
GROUNDED_ANSWER = Prompt(
    name="grounded_answer",
    system=(
        "You answer questions about the user's documents using only the "
        "numbered excerpts supplied with each question.\n\n"
        "Cite the excerpt behind every factual claim using its bracketed "
        "number, like [1] or [2][3], placed immediately after the claim. A "
        "reader must be able to verify each statement from the citation.\n\n"
        "Use only what the excerpts contain. If they do not answer the "
        "question, say plainly that the documents do not cover it and describe "
        "what they do address instead. That is a correct, useful answer - do "
        "not fill the gap with outside knowledge or inference.\n\n"
        "When excerpts disagree, present both and cite each. When the answer is "
        "partial, answer what you can and name what is missing.\n\n"
        "Write for a colleague: lead with the answer, keep it as short as the "
        "question allows, and use the document's own terms."
    ),
)

# Suggestions are offered to the user as clickable prompts, so they must be
# answerable from the same material - suggesting a question the vault cannot
# answer trains users to distrust the feature.
FOLLOW_UP_SUGGESTIONS = Prompt(
    name="follow_up_suggestions",
    system=(
        "You propose follow-up questions a reader would naturally ask next.\n\n"
        "Every question must be answerable from the supplied excerpts, specific "
        "rather than generic, and phrased as the user would type it. Avoid "
        "repeating what the previous answer already covered."
    ),
)

SESSION_TITLE = Prompt(
    name="session_title",
    system=(
        "You write a short title for a conversation, in four to six words, "
        "naming its subject. Use plain nouns, no quotes and no trailing "
        "punctuation. Return only the title."
    ),
)


def build_condense_messages(
    question: str, history: Sequence[tuple[str, str]]
) -> list[ChatMessage]:
    """Messages that rewrite a follow-up into a standalone question."""
    transcript = "\n".join(
        f"{'User' if role == USER else 'Assistant'}: {content}"
        for role, content in history
    )
    return QUESTION_CONDENSER.build(
        f"Conversation so far:\n{transcript}\n\nFollow-up question: {question}"
    )


def build_answer_messages(
    question: str,
    *,
    context_block: str,
    history: Sequence[tuple[str, str]] = (),
    caveat: str = "",
) -> list[ChatMessage]:
    """Messages for answering a question from retrieved excerpts.

    Prior turns are included as real conversation messages rather than being
    flattened into the prompt, so the model distinguishes what was previously
    said from the evidence it must cite.
    """
    messages = [ChatMessage(role=SYSTEM, content=GROUNDED_ANSWER.system + caveat)]
    messages.extend(
        ChatMessage(role=role, content=content) for role, content in history
    )
    messages.append(
        ChatMessage(
            role=USER,
            content=(
                f"Excerpts from my documents:\n\n{context_block}\n\n"
                f"Question: {question}"
            ),
        )
    )
    return messages


def build_suggestion_messages(
    question: str, answer: str, *, context_block: str, count: int
) -> list[ChatMessage]:
    """Messages for proposing follow-up questions."""
    return FOLLOW_UP_SUGGESTIONS.build(
        f"Excerpts:\n\n{context_block}\n\n"
        f"The user asked: {question}\n\n"
        f"You answered: {answer}\n\n"
        f"Propose {count} follow-up questions."
    )


def build_title_messages(question: str, answer: str) -> list[ChatMessage]:
    """Messages for naming a chat session from its opening exchange."""
    return SESSION_TITLE.build(f"Question: {question}\n\nAnswer: {answer}")


# --------------------------------------------------------------------------- #
# Conversation questions
# --------------------------------------------------------------------------- #
# Questions about the exchange itself ("what did I ask?", "where did that
# excerpt come from?") are answerable from the transcript, so routing them
# through document retrieval only produces a confusing refusal. This prompt
# answers from the transcript and nothing else.
CONVERSATION_ANSWER = Prompt(
    name="conversation_answer",
    system=(
        "You answer questions about the conversation you are part of, using "
        "the transcript supplied and nothing else.\n\n"
        "Quote or restate what was actually said. If the transcript does not "
        "contain the answer, say so plainly.\n\n"
        "Do not consult or speculate about the user's documents here - this "
        "question is about the exchange, not about their content."
    ),
)

# Counting and listing questions cannot be answered safely from retrieved
# passages, because retrieval returns a sample rather than the whole document.
# The instruction makes the limit of the evidence explicit instead of letting a
# confident-sounding total stand unqualified.
AGGREGATE_CAVEAT = (
    "\n\nThis question asks for a count, total or complete list. The excerpts "
    "below are a selection from the documents, not the whole of them, so state "
    "plainly that your answer covers only what is shown and may be incomplete. "
    "Give the figure you can support from these excerpts rather than estimating "
    "the true total."
)


def build_conversation_messages(
    question: str, history: Sequence[tuple[str, str]]
) -> list[ChatMessage]:
    """Messages for answering a question about the conversation itself."""
    transcript = "\n\n".join(
        f"{'User' if role == USER else 'Assistant'}: {content}"
        for role, content in history
    )
    return CONVERSATION_ANSWER.build(
        f"Transcript of the conversation so far:\n\n{transcript}\n\n"
        f"Question about this conversation: {question}"
    )

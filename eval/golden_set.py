"""Hand-labelled retrieval ground truth.

Each case is a question written the way a user would actually ask it - plain
language, not document vocabulary - paired with the chunk indices that answer
it. Labels were read off the source document by hand.

Keeping questions in user language is the point: questions phrased in the
document's own words are trivially retrievable by either arm and tell you
nothing about whether retrieval works.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One labelled question."""

    question: str
    #: Chunk indices that answer it. A retrieval is a hit if any is returned.
    gold_chunks: frozenset[int]
    #: False for questions the corpus cannot answer; these must be refused.
    answerable: bool = True


#: SBI General health insurance policy, 76 chunks.
SBI_POLICY = uuid.UUID("9bdfcc22-8e74-457c-9013-95d5f967423e")

SBI_POLICY_CASES: tuple[GoldenCase, ...] = (
    GoldenCase("can I be treated at home instead of staying in a ward?", frozenset({5, 6})),
    GoldenCase("does my plan pay for a bed in a private room?", frozenset({16, 17})),
    GoldenCase("how long do I have to wait before an old condition is paid for?", frozenset({23})),
    GoldenCase("will you pay someone to look after my elderly mother after she leaves?", frozenset({21})),
    GoldenCase("is there help with costs for my kid staying overnight?", frozenset({22})),
    GoldenCase("what about alternative medicine like homeopathy?", frozenset({19})),
    GoldenCase("do you cover the person who donates an organ?", frozenset({20})),
    GoldenCase("are costs before I am admitted reimbursed?", frozenset({18})),
    GoldenCase("what if I switch from another company's plan?", frozenset({24, 25, 27, 28})),
    GoldenCase("is a day procedure that does not need an overnight stay included?", frozenset({4})),
    GoldenCase("how much of the bill do I have to pay myself?", frozenset({7})),
    # Controls: nothing in an insurance policy answers these.
    GoldenCase("Who won the football world cup in 1998?", frozenset(), answerable=False),
    GoldenCase("What is the capital city of Brazil?", frozenset(), answerable=False),
)

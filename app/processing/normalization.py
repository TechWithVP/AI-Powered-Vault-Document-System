"""Text normalisation applied after extraction.

Some PDFs simulate bold type by overprinting the same glyph several times.
Extraction then yields runs like "SSSSBBBBIIII" for "SBI", which waste tokens,
pollute citations, and - because the same artefact prefixes every chunk - make
chunks embed more similarly to one another, blunting retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby

from app.core.logging import get_logger

logger = get_logger(__name__)

#: A token must contain at least this many distinct character groups before a
#: uniform repeat is treated as overprinting rather than ordinary spelling.
MIN_GROUPS = 3

#: Smallest repeat count considered artificial. Doubling is the common case
#: after quadrupling.
MIN_REPEAT = 2

#: Collapses runs of three or more blank lines left behind by stripped artefacts.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

#: Trailing whitespace on a line adds nothing and inflates token counts.
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class NormalisationReport:
    """What a normalisation pass changed, for logging and tests."""

    tokens_collapsed: int = 0
    characters_removed: int = 0

    @property
    def changed(self) -> bool:
        return self.tokens_collapsed > 0 or self.characters_removed > 0


class TextNormalizer:
    """Repairs extraction artefacts without altering legitimate text."""

    def collapse_overprinted(self, token: str) -> str:
        """Collapse a token whose every character is repeated the same number of times.

        "SSSSBBBBIIII" becomes "SBI" because S, B and I each repeat exactly four
        times. Ordinary words fail the uniformity test - "committee" has runs of
        1, 1, 2, 1, 2 and 2 - so they are left untouched. Purely numeric tokens
        are skipped outright: a phone number such as "9900112233" is uniformly
        doubled yet entirely legitimate.
        """
        if not token.isalpha() or len(token) < MIN_GROUPS * MIN_REPEAT:
            return token

        groups = [(character, len(list(run))) for character, run in groupby(token)]
        if len(groups) < MIN_GROUPS:
            return token

        repeat = groups[0][1]
        if repeat < MIN_REPEAT or any(count != repeat for _, count in groups):
            return token

        return "".join(character for character, _ in groups)

    def normalize_text(self, text: str) -> tuple[str, NormalisationReport]:
        """Return the cleaned text plus a report of what changed."""
        collapsed = 0
        pieces: list[str] = []

        # Split on whitespace but keep it, so layout and line breaks survive.
        for piece in re.split(r"(\s+)", text):
            if piece.isspace() or not piece:
                pieces.append(piece)
                continue
            repaired = self.collapse_overprinted(piece)
            if repaired != piece:
                collapsed += 1
            pieces.append(repaired)

        cleaned = "".join(pieces)
        cleaned = _TRAILING_SPACE.sub("", cleaned)
        cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)

        return cleaned, NormalisationReport(
            tokens_collapsed=collapsed,
            characters_removed=max(len(text) - len(cleaned), 0),
        )

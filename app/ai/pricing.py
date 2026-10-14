"""AI cost calculation.

Published per-million-token rates, kept in one place so cost reporting has a
single source of truth. Unknown models price at zero rather than failing: a
missing rate must never break an ingestion run.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)

PER_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class ModelRate:
    """USD per million tokens for one model."""

    input_usd: Decimal
    output_usd: Decimal = Decimal("0")


#: Rates as published by OpenAI (USD per 1M tokens).
MODEL_RATES: dict[str, ModelRate] = {
    "gpt-4o-mini": ModelRate(Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": ModelRate(Decimal("2.50"), Decimal("10.00")),
    "gpt-4.1-mini": ModelRate(Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1": ModelRate(Decimal("2.00"), Decimal("8.00")),
    "text-embedding-3-small": ModelRate(Decimal("0.02")),
    "text-embedding-3-large": ModelRate(Decimal("0.13")),
}


class CostCalculator:
    """Converts token counts into USD."""

    def __init__(self, rates: dict[str, ModelRate] | None = None) -> None:
        self._rates = rates or MODEL_RATES

    def rate_for(self, model: str) -> ModelRate | None:
        """Resolve a rate, tolerating dated model aliases such as -2024-07-18."""
        if model in self._rates:
            return self._rates[model]
        for known, rate in self._rates.items():
            if model.startswith(known):
                return rate
        return None

    def calculate(
        self, model: str, *, prompt_tokens: int, completion_tokens: int = 0
    ) -> Decimal:
        rate = self.rate_for(model)
        if rate is None:
            logger.warning("No published rate for model", extra={"model": model})
            return Decimal("0")

        input_cost = Decimal(prompt_tokens) * rate.input_usd / PER_MILLION
        output_cost = Decimal(completion_tokens) * rate.output_usd / PER_MILLION
        return (input_cost + output_cost).quantize(Decimal("0.000001"))


#: Shared calculator; rates are static so a single instance is sufficient.
cost_calculator = CostCalculator()

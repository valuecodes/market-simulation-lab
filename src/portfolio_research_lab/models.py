"""Pydantic configuration models.

These models describe *what* to simulate. They are deliberately free of any
pandas or simulation logic so that a strategy configuration can be created,
validated, serialised to JSON and shared without touching the engine.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator, model_validator

# Assets whose combined weights fall within this tolerance of 1.0 are treated
# as fully invested. Anything outside is rejected to catch typos early.
WEIGHT_TOLERANCE = 1e-6

# Rebalancing cadences the simulator understands. ``None`` means buy-and-hold
# (weights are set once and left to drift).
REBALANCE_FREQUENCIES = frozenset({"monthly", "quarterly", "annually"})


class StrategyConfig(BaseModel):
    """Reusable description of a portfolio strategy.

    Attributes
    ----------
    name:
        Human-readable label shown in the UI and charts.
    initial_capital:
        Starting cash, in the currency of the price data.
    allocations:
        Mapping of asset symbol to target weight. Weights must be positive and
        sum to 1.0 (small floating-point error is tolerated).
    benchmark:
        Optional symbol used as a comparison baseline. It does not need to be
        part of ``allocations``.
    trading_days_per_year:
        Periods used to annualise returns and volatility. Defaults to 252 for
        daily data; use 12 for monthly data.
    rebalance_frequency:
        How often to reset holdings back to ``allocations``. ``None`` (the
        default) is buy-and-hold: weights are set once and drift. Otherwise one
        of ``"monthly"``, ``"quarterly"`` or ``"annually"``.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(default="Buy & Hold", min_length=1)
    initial_capital: float = Field(default=10_000.0, gt=0)
    allocations: dict[str, float] = Field(min_length=1)
    benchmark: str | None = None
    trading_days_per_year: int = Field(default=252, gt=0)
    rebalance_frequency: str | None = None

    @field_validator("rebalance_frequency")
    @classmethod
    def _known_rebalance_frequency(cls, value: str | None) -> str | None:
        if value is not None and value not in REBALANCE_FREQUENCIES:
            allowed = ", ".join(sorted(REBALANCE_FREQUENCIES))
            raise ValueError(
                f"rebalance_frequency must be one of {{{allowed}}} or None, got {value!r}"
            )
        return value

    @field_validator("allocations")
    @classmethod
    def _weights_must_be_positive(cls, value: dict[str, float]) -> dict[str, float]:
        for symbol, weight in value.items():
            if weight <= 0:
                raise ValueError(f"weight for {symbol!r} must be positive, got {weight}")
        return value

    @model_validator(mode="after")
    def _weights_must_sum_to_one(self) -> StrategyConfig:
        total = sum(self.allocations.values())
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"allocation weights must sum to 1.0, got {total:.6f}")
        return self

    @property
    def symbols(self) -> list[str]:
        """Asset symbols in a stable order."""
        return list(self.allocations)

    @classmethod
    def from_weights(
        cls,
        allocations: Mapping[str, float],
        *,
        normalize: bool = False,
        name: str = "Buy & Hold",
        initial_capital: float = 10_000.0,
        benchmark: str | None = None,
        trading_days_per_year: int = 252,
        rebalance_frequency: str | None = None,
    ) -> StrategyConfig:
        """Build a config, optionally normalising raw weights to sum to 1.0.

        Handy for UIs where the user enters percentages or shares (integers are
        fine) that do not already add up to one.
        """
        weights: dict[str, float] = dict(allocations)
        if normalize:
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("cannot normalize allocations that sum to <= 0")
            weights = {symbol: weight / total for symbol, weight in weights.items()}
        return cls(
            allocations=weights,
            name=name,
            initial_capital=initial_capital,
            benchmark=benchmark,
            trading_days_per_year=trading_days_per_year,
            rebalance_frequency=rebalance_frequency,
        )

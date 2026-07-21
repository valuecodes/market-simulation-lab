"""Pydantic configuration models.

These models describe *what* to simulate. They are deliberately free of any
pandas or simulation logic so that a strategy configuration can be created,
validated, serialised to JSON and shared without touching the engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise

from pydantic import BaseModel, Field, field_validator, model_validator

# Assets whose combined weights fall within this tolerance of 1.0 are treated
# as fully invested. Anything outside is rejected to catch typos early.
WEIGHT_TOLERANCE = 1e-6

# Rebalancing cadences the simulator understands. ``None`` means buy-and-hold
# (weights are set once and left to drift).
REBALANCE_FREQUENCIES = frozenset({"monthly", "quarterly", "annually"})

# Moving-average kinds the trend-timing engine understands.
MA_KINDS = frozenset({"simple", "exponential"})


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
    initial_capital: float = Field(default=10_000.0, gt=0, allow_inf_nan=False)
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
            # NaN slips past ``weight <= 0`` (all NaN comparisons are False) and
            # would then bypass the sum check too, so reject non-finite explicitly.
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"weight for {symbol!r} must be a positive number, got {weight}")
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


class DeployRule(BaseModel):
    """A tactical cash-deployment rule for the cash-deploy backtest.

    A rule splits a cash reserve into tranches keyed to market-drawdown depth:
    when the market draws down past each ``thresholds[k]`` (a positive fraction,
    so ``0.10`` means a 10% fall from the running peak), the fraction
    ``usages[k]`` of the reserve is deployed into stocks.

    Attributes
    ----------
    name:
        Human-readable label shown in the UI.
    thresholds:
        Drawdown magnitudes as positive fractions, strictly increasing, each in
        ``(0, 1)``. E.g. ``(0.10, 0.20, 0.30, 0.40, 0.50)``.
    usages:
        Fraction of the reserve to deploy at each matching threshold. Same length
        as ``thresholds`` and each positive. Usages summing to 1.0 mean the whole
        reserve is deployed by the deepest threshold, but this is *not* enforced:
        a sum below 1.0 leaves a permanent cash residue, and a sum above 1.0 is
        capped by the cash actually available at run time.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1)
    thresholds: tuple[float, ...] = Field(min_length=1)
    usages: tuple[float, ...] = Field(min_length=1)

    @field_validator("thresholds")
    @classmethod
    def _thresholds_increasing_in_unit_interval(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for threshold in value:
            if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
                raise ValueError(f"each threshold must be a number in (0, 1), got {threshold}")
        if any(b <= a for a, b in pairwise(value)):
            raise ValueError(f"thresholds must be strictly increasing, got {value}")
        return value

    @field_validator("usages")
    @classmethod
    def _usages_must_be_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for usage in value:
            if not math.isfinite(usage) or usage <= 0:
                raise ValueError(f"each reserve usage must be a positive number, got {usage}")
        return value

    @model_validator(mode="after")
    def _thresholds_and_usages_align(self) -> DeployRule:
        if len(self.thresholds) != len(self.usages):
            raise ValueError(
                f"thresholds and usages must have equal length, got "
                f"{len(self.thresholds)} and {len(self.usages)}"
            )
        return self

    @property
    def usage_sum(self) -> float:
        """Total reserve fraction deployed if every threshold is reached."""
        return sum(self.usages)


class CashDeployConfig(BaseModel):
    """Configuration for a tactical cash-deployment backtest.

    Start with ``reserve_pct`` of capital in cash (a money-market account) and
    the rest in stocks. As the market draws down, deploy the reserve into stocks
    per ``rule``; once the market recovers to a new high, refill the reserve
    toward its target by drip-selling stocks at ``refill_rate_per_year`` of the
    target per year.

    Attributes
    ----------
    name:
        Human-readable label shown in the UI and charts.
    initial_capital:
        Starting capital, in the currency of the price data.
    reserve_pct:
        Cash fraction at inception, in ``[0, 1]``. Stocks get ``1 - reserve_pct``.
    rule:
        The :class:`DeployRule` that drives deployment.
    refill_rate_per_year:
        Fraction of the *target* reserve moved back from stocks to cash per year,
        while at a new all-time high. ``0.25`` refills a fully-drained reserve in
        roughly four years.
    trading_days_per_year:
        Periods used to annualise the refill drip and metrics (252 for daily).
    stock_symbol / cash_symbol:
        Column names of the stock and cash legs in the price frame passed to the
        engine.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(default="Cash Deploy", min_length=1)
    initial_capital: float = Field(default=10_000.0, gt=0, allow_inf_nan=False)
    reserve_pct: float = Field(default=0.30, ge=0.0, le=1.0)
    rule: DeployRule
    refill_rate_per_year: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    trading_days_per_year: int = Field(default=252, gt=0)
    stock_symbol: str = Field(default="S&P 500", min_length=1)
    cash_symbol: str = Field(default="Cash (Fed Funds)", min_length=1)

    @model_validator(mode="after")
    def _symbols_must_differ(self) -> CashDeployConfig:
        if self.stock_symbol == self.cash_symbol:
            raise ValueError("stock_symbol and cash_symbol must be different")
        return self


class TimingConfig(BaseModel):
    """Configuration for a moving-average trend-timing backtest.

    Hold 100% stocks while the stock price is above its ``ma_window`` moving
    average, and 100% cash (a money-market account) when it falls below. The
    signal is evaluated on the previous close and acted on the next day (the
    engine lags it one bar to avoid look-ahead).

    Attributes
    ----------
    name:
        Human-readable label shown in the UI and charts.
    initial_capital:
        Starting capital, in the currency of the price data.
    ma_window:
        Moving-average lookback in trading days (e.g. ``200``). Must be > 1.
    ma_kind:
        ``"simple"`` (equal-weighted rolling mean) or ``"exponential"`` (EWM with
        ``span = ma_window``). Both stay fully invested until the average is
        defined (the first ``ma_window - 1`` steps).
    band_pct:
        Hysteresis buffer as a positive fraction in ``[0, 1)``. Exit to cash only
        when ``price < ma * (1 - band_pct)`` and re-enter only when
        ``price > ma * (1 + band_pct)``; inside the band the position is retained,
        which damps whipsaws around the crossover.
    cost_bps:
        Transaction cost charged on each switch, in basis points of the traded
        notional (``10`` = 0.10%). Bounded below 10,000 bps so a switch can never
        wipe out or invert a leg. Defaults to 0 for parity with the other engines.
    trading_days_per_year:
        Periods used to annualise metrics (252 for daily data).
    stock_symbol / cash_symbol:
        Column names of the stock and cash legs in the price frame.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(default="Trend Timing", min_length=1)
    initial_capital: float = Field(default=10_000.0, gt=0, allow_inf_nan=False)
    ma_window: int = Field(default=200, gt=1)
    ma_kind: str = Field(default="simple")
    band_pct: float = Field(default=0.0, ge=0.0, lt=1.0, allow_inf_nan=False)
    cost_bps: float = Field(default=0.0, ge=0.0, lt=10_000.0, allow_inf_nan=False)
    trading_days_per_year: int = Field(default=252, gt=0)
    stock_symbol: str = Field(default="S&P 500", min_length=1)
    cash_symbol: str = Field(default="Cash (Fed Funds)", min_length=1)

    @field_validator("ma_kind")
    @classmethod
    def _known_ma_kind(cls, value: str) -> str:
        if value not in MA_KINDS:
            allowed = ", ".join(sorted(MA_KINDS))
            raise ValueError(f"ma_kind must be one of {{{allowed}}}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _symbols_must_differ(self) -> TimingConfig:
        if self.stock_symbol == self.cash_symbol:
            raise ValueError("stock_symbol and cash_symbol must be different")
        return self


# The named deploy rules from the research brief. Thresholds and usages are
# expressed as fractions (10% -> 0.10).
PRESET_RULES: dict[str, DeployRule] = {
    "User rule": DeployRule(
        name="User rule",
        thresholds=(0.10, 0.20, 0.30, 0.40, 0.50),
        usages=(0.15, 0.30, 0.30, 0.20, 0.05),
    ),
    "Growth optimum": DeployRule(
        name="Growth optimum",
        thresholds=(0.10, 0.15, 0.20, 0.30, 0.40),
        usages=(0.30, 0.25, 0.20, 0.15, 0.10),
    ),
    "Risk optimum": DeployRule(
        name="Risk optimum",
        thresholds=(0.15, 0.20, 0.30, 0.40, 0.50),
        usages=(0.10, 0.15, 0.20, 0.25, 0.30),
    ),
    "Recommended": DeployRule(
        name="Recommended",
        thresholds=(0.15, 0.20, 0.30, 0.40, 0.50),
        usages=(0.30, 0.25, 0.20, 0.15, 0.10),
    ),
}

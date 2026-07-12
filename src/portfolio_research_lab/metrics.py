"""Performance metrics.

Every function here is pure: it takes pandas objects in and returns numbers or
pandas objects out, with no dependency on the simulator. This makes the metrics
easy to unit-test in isolation and reuse against any equity curve.

An "equity curve" is a :class:`pandas.Series` of portfolio value over time,
indexed by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def periodic_returns(equity: pd.Series) -> pd.Series:
    """Simple period-over-period returns of an equity curve.

    The first period has no prior value, so it is dropped.
    """
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    """Total return over the whole series: ``end / start - 1``."""
    _require_two_points(equity)
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate.

    Uses the number of *periods* in the series (not calendar dates) to derive
    the number of years, which keeps the metric consistent with synthetic data.
    """
    _require_two_points(equity)
    num_periods = len(equity) - 1
    years = num_periods / periods_per_year
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised standard deviation of periodic returns (sample std)."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown at each point: ``value / running_peak - 1`` (values <= 0)."""
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.25)."""
    if equity.empty:
        return 0.0
    return float(drawdown_series(equity).min())


def summarize(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Convenience bundle of the headline metrics for an equity curve."""
    rets = periodic_returns(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "max_drawdown": max_drawdown(equity),
    }


def _require_two_points(equity: pd.Series) -> None:
    if len(equity) < 2:
        raise ValueError("equity curve needs at least two points")

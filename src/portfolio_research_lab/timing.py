"""The moving-average trend-timing engine.

Holds 100% stocks while the stock price is above its moving average and moves
100% to cash (a money-market account) when it falls below. Like
:mod:`~portfolio_research_lab.cash_deploy` this is *path dependent* — the position
held on any day depends on the prior signal and the running in/out state — so it
is a dedicated explicit daily loop rather than a fixed-weight strategy plugged
into the shared simulator.

The signal is lagged one bar to avoid look-ahead: the position held *during* day
``i`` (and therefore earning day ``i``'s return) is decided from the previous
close (``price[i-1]`` versus ``ma[i-1]``); any resulting switch executes at day
``i``'s close. A hysteresis ``band`` keeps the position unchanged while the price
sits within ``±band`` of the average, damping whipsaws.

The engine consumes the same two-column ``[stock, cash]`` price frame the app
builds (see :func:`portfolio_research_lab.data.load_stocks_cash`). Cash held out
of the market accrues at the money-market rate, taken as the ratio of consecutive
cash-index levels — identical to the cash-deploy engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_research_lab import metrics
from portfolio_research_lab.models import TimingConfig

# Columns of the switch-event log emitted by a run.
_EVENT_COLUMNS = ("type", "price", "ma", "cost", "value")


@dataclass(slots=True)
class TimingResult:
    """Output of a moving-average trend-timing run.

    Attributes
    ----------
    config:
        The configuration that produced this run.
    equity:
        Total portfolio value over time (stock leg + cash).
    stock_value:
        Market value of the stock leg over time (0 while out of the market).
    cash:
        Cash balance over time (0 while invested).
    position:
        Exposure held *during* each day, as 1.0 (in stocks) or 0.0 (in cash).
        Recorded before any same-day switch, so ``position[i]`` is the exposure
        that earned day ``i``'s return.
    moving_average:
        The moving average of the stock price (NaN during the warm-up window).
    events:
        One row per switch, indexed by date, with columns ``type``
        (``"to_stocks"``/``"to_cash"``), ``price`` (execution close), ``ma`` (the
        prior-day average that triggered the switch), ``cost`` (cash lost to the
        transaction cost) and ``value`` (portfolio value just after the switch).
    """

    config: TimingConfig
    equity: pd.Series
    stock_value: pd.Series
    cash: pd.Series
    position: pd.Series
    moving_average: pd.Series
    events: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        """Headline metrics for the strategy equity curve."""
        return metrics.summarize(self.equity, self.config.trading_days_per_year)

    def drawdown(self) -> pd.Series:
        """Drawdown series of the strategy equity curve."""
        return metrics.drawdown_series(self.equity)

    @property
    def n_switches(self) -> int:
        """Number of in/out switches over the run."""
        return len(self.events)

    @property
    def time_in_market(self) -> float:
        """Fraction of return intervals spent invested in stocks.

        Averaged over the ``n-1`` return intervals (the first observation has no
        preceding return), so it reflects realised exposure rather than a raw row
        count.
        """
        if len(self.position) < 2:
            return float(self.position.iloc[0]) if len(self.position) else 0.0
        return float(self.position.iloc[1:].mean())


def _moving_average(stock: np.ndarray, config: TimingConfig) -> np.ndarray:
    """Moving average of the stock price, NaN for the first ``ma_window-1`` steps.

    Both kinds share the same warm-up mask so the two modes initialise
    identically: the strategy stays fully invested until the average is defined.
    """
    series = pd.Series(stock)
    window = config.ma_window
    if config.ma_kind == "exponential":
        ma = series.ewm(span=window, adjust=False).mean()
        # ``ewm`` is defined from the first row; mask the warm-up so it matches the
        # simple average's ``min_periods=window`` behaviour.
        ma.iloc[: window - 1] = np.nan
    else:
        ma = series.rolling(window, min_periods=window).mean()
    return ma.to_numpy(dtype=float)


def run_ma_timing(prices: pd.DataFrame, config: TimingConfig) -> TimingResult:
    """Run a moving-average trend-timing simulation.

    Parameters
    ----------
    prices:
        Wide-format price frame containing (at least) ``config.stock_symbol`` and
        ``config.cash_symbol`` columns, indexed by a sorted date index. The cash
        column is a money-market growth index; its per-step ratio drives interest
        on cash held out of the market.
    config:
        The trend-timing configuration to simulate.
    """
    if prices.empty:
        raise ValueError("cannot simulate on empty price data")
    if len(prices) < 2:
        raise ValueError("need at least two price rows to simulate")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("price data must be sorted by a monotonically increasing index")

    missing = [s for s in (config.stock_symbol, config.cash_symbol) if s not in prices.columns]
    if missing:
        raise ValueError(f"price data is missing required columns: {missing}")

    index = prices.index
    stock = prices[config.stock_symbol].to_numpy(dtype=float)
    cash_level = prices[config.cash_symbol].to_numpy(dtype=float)

    if (stock <= 0.0).any() or not np.isfinite(stock).all():
        bad = stock[~np.isfinite(stock) | (stock <= 0.0)][0]
        raise ValueError(f"stock prices must be finite and positive, got {bad}")
    if (cash_level <= 0.0).any() or not np.isfinite(cash_level).all():
        bad = cash_level[~np.isfinite(cash_level) | (cash_level <= 0.0)][0]
        raise ValueError(f"cash levels must be finite and positive, got {bad}")

    # Per-step cash growth factor = ratio of consecutive money-market levels; the
    # first step has no prior level so it is 1.0. Mirrors the cash-deploy engine.
    factor = np.ones(len(stock), dtype=float)
    if len(stock) > 1:
        prev = cash_level[:-1]
        ratio = np.divide(
            cash_level[1:],
            prev,
            out=np.ones(len(stock) - 1, dtype=float),
            where=(prev > 0.0),
        )
        ratio[~np.isfinite(ratio)] = 1.0
        factor[1:] = ratio

    ma = _moving_average(stock, config)
    band = config.band_pct
    cost_rate = config.cost_bps / 10_000.0

    # --- State ---
    invested = True  # start fully invested; the MA is undefined during warm-up
    units = config.initial_capital / stock[0]
    cash = 0.0

    n = len(index)
    equity_out = np.empty(n, dtype=float)
    cash_out = np.empty(n, dtype=float)
    stock_out = np.empty(n, dtype=float)
    position_out = np.empty(n, dtype=float)
    events: list[tuple[pd.Timestamp, str, float, float, float, float]] = []

    for i in range(n):
        price = stock[i]
        if i > 0:
            cash *= factor[i]  # accrue money-market yield on any idle cash

        # The exposure that earns day i's return is the state entering the day,
        # before any switch executed today (which only affects tomorrow's return).
        position_out[i] = 1.0 if invested else 0.0

        # Decide from the PREVIOUS close (no look-ahead); execute at today's close.
        # Transitions are state-dependent with a hysteresis band: exit only below
        # the lower boundary, enter only above the upper one, otherwise hold — so a
        # flat price at the average (band 0) never oscillates.
        if i >= 1 and np.isfinite(ma[i - 1]):
            prev_price = stock[i - 1]
            prev_ma = ma[i - 1]
            if invested and prev_price < prev_ma * (1.0 - band):
                notional = units * price  # stocks -> cash
                cost = cost_rate * notional
                cash = notional - cost
                units = 0.0
                invested = False
                events.append((index[i], "to_cash", price, prev_ma, cost, cash))
            elif not invested and prev_price > prev_ma * (1.0 + band):
                notional = cash  # cash -> stocks
                cost = cost_rate * notional
                units = (cash - cost) / price
                cash = 0.0
                invested = True
                events.append((index[i], "to_stocks", price, prev_ma, cost, units * price))

        equity_out[i] = cash + units * price
        cash_out[i] = cash
        stock_out[i] = units * price

    events_frame = pd.DataFrame(
        [e[1:] for e in events],
        index=pd.DatetimeIndex([e[0] for e in events], name=index.name),
        columns=list(_EVENT_COLUMNS),
    )

    return TimingResult(
        config=config,
        equity=pd.Series(equity_out, index=index, name="equity"),
        stock_value=pd.Series(stock_out, index=index, name=config.stock_symbol),
        cash=pd.Series(cash_out, index=index, name=config.cash_symbol),
        position=pd.Series(position_out, index=index, name="position"),
        moving_average=pd.Series(ma, index=index, name="moving_average"),
        events=events_frame,
    )

"""The tactical cash-deployment engine.

Unlike the fixed-weight :mod:`~portfolio_research_lab.simulator` engine, this
simulation is *path dependent*: it holds a cash reserve and deploys it into
stocks as the market draws down from its running peak, then refills the reserve
by drip-selling stocks once a new all-time high is reached. That state (running
peak, which tranches have fired this episode, the locked tranche base) cannot be
expressed as a per-period weight vector, so this is a dedicated explicit daily
loop rather than a strategy plugged into the shared simulator.

The engine consumes the same two-column ``[stock, cash]`` price frame the app
builds (see :func:`portfolio_research_lab.data.load_stocks_cash`). The cash leg
is a money-market growth index; cash held in the reserve accrues at the same
per-step rate, taken as the ratio of consecutive cash-index levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_research_lab import metrics
from portfolio_research_lab.models import CashDeployConfig

# Columns of the event log emitted by a run.
_EVENT_COLUMNS = ("type", "drawdown", "amount", "cash_after")


@dataclass(slots=True)
class CashDeployResult:
    """Output of a cash-deploy simulation run.

    Attributes
    ----------
    config:
        The configuration that produced this run.
    equity:
        Total portfolio value over time (cash + stock leg).
    cash:
        Cash reserve balance over time.
    stock_value:
        Market value of the stock leg over time.
    events:
        One row per deploy/refill action, indexed by date, with columns
        ``type`` (``"deploy"``/``"refill"``), ``drawdown`` (the S&P drawdown at
        the time), ``amount`` (cash moved) and ``cash_after`` (reserve balance
        after the action).
    """

    config: CashDeployConfig
    equity: pd.Series
    cash: pd.Series
    stock_value: pd.Series
    events: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        """Headline metrics for the strategy equity curve."""
        return metrics.summarize(self.equity, self.config.trading_days_per_year)

    def drawdown(self) -> pd.Series:
        """Drawdown series of the strategy equity curve."""
        return metrics.drawdown_series(self.equity)


def run_cash_deploy(prices: pd.DataFrame, config: CashDeployConfig) -> CashDeployResult:
    """Run a tactical cash-deployment simulation.

    Parameters
    ----------
    prices:
        Wide-format price frame containing (at least) ``config.stock_symbol`` and
        ``config.cash_symbol`` columns. The cash column is a money-market growth
        index; its per-step ratio drives interest on the held reserve.
    config:
        The cash-deploy configuration to simulate.
    """
    if prices.empty:
        raise ValueError("cannot simulate on empty price data")

    missing = [s for s in (config.stock_symbol, config.cash_symbol) if s not in prices.columns]
    if missing:
        raise ValueError(f"price data is missing required columns: {missing}")

    index = prices.index
    stock = prices[config.stock_symbol].to_numpy(dtype=float)
    cash_level = prices[config.cash_symbol].to_numpy(dtype=float)

    if stock[0] <= 0.0 or not np.isfinite(stock[0]):
        raise ValueError(f"the first stock price must be positive, got {stock[0]}")

    # Per-step cash growth factor = ratio of consecutive money-market levels; the
    # first step has no prior level so it is 1.0. A non-positive/non-finite prior
    # level would corrupt the ratio, so guard it (falls back to no growth).
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

    thresholds = config.rule.thresholds
    usages = config.rule.usages
    r = config.reserve_pct
    refill_daily = config.refill_rate_per_year / config.trading_days_per_year

    # --- State ---
    cash = config.initial_capital * r
    units = config.initial_capital * (1.0 - r) / stock[0]
    peak = stock[0]
    deployed = [False] * len(thresholds)  # which tranches have fired this episode
    base: float | None = None  # reserve locked at the first dip of an episode

    n = len(index)
    equity_out = np.empty(n, dtype=float)
    cash_out = np.empty(n, dtype=float)
    stock_out = np.empty(n, dtype=float)
    events: list[tuple[pd.Timestamp, str, float, float, float]] = []

    for i in range(n):
        price = stock[i]
        if i > 0:
            cash *= factor[i]  # accrue the reserve's money-market yield

        if price >= peak:
            # At or making a new all-time high: the drawdown episode is over.
            peak = price
            deployed = [False] * len(thresholds)
            base = None
            # Refill the reserve toward its target by drip-selling stocks. The
            # target is a share of the *current* portfolio, so this also trims
            # winners at new highs to hold the reserve at `reserve_pct` — not
            # only after a deployment. It is therefore a slow rebalance toward
            # the target allocation, accelerated on the buy side by drawdown
            # deployment. Gating on price >= peak is the literal "after back to
            # ATH" reading; on a long plateau just below the peak the refill
            # stalls. To relax that, widen to a band like `price >= peak*(1-b)`.
            stock_value = units * price
            target_cash = r * (cash + stock_value)
            if cash < target_cash:
                move = min(refill_daily * target_cash, target_cash - cash, stock_value)
                if move > 0.0:
                    units -= move / price
                    cash += move
                    events.append((index[i], "refill", 0.0, move, cash))
        else:
            drawdown = price / peak - 1.0  # negative
            if base is None:
                base = cash  # lock the tranche base at the episode's first dip
            for k, threshold in enumerate(thresholds):
                # Every threshold newly crossed this step fires, so a single
                # gap-down day can deploy several tranches at once.
                if not deployed[k] and -drawdown >= threshold:
                    amount = min(base * usages[k], cash)
                    if amount > 0.0:
                        units += amount / price
                        cash -= amount
                        events.append((index[i], "deploy", drawdown, amount, cash))
                    deployed[k] = True

        equity_out[i] = cash + units * price
        cash_out[i] = cash
        stock_out[i] = units * price

    events_frame = pd.DataFrame(
        [e[1:] for e in events],
        index=pd.DatetimeIndex([e[0] for e in events], name=index.name),
        columns=list(_EVENT_COLUMNS),
    )

    return CashDeployResult(
        config=config,
        equity=pd.Series(equity_out, index=index, name="equity"),
        cash=pd.Series(cash_out, index=index, name=config.cash_symbol),
        stock_value=pd.Series(stock_out, index=index, name=config.stock_symbol),
        events=events_frame,
    )

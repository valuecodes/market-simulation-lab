"""Cash Deploy — a tactical cash-deployment backtest page.

Holds part of the portfolio in a cash reserve (money-market at the fed funds
rate), deploys it into stocks in tranches as the S&P draws down from its
all-time high, and drips it back to target once the market recovers. Compares
the result against a 100%-stocks buy-and-hold and a static stock/cash split so
the user can see whether tactical timing actually helped.

Like ``Home.py`` this is a thin presentation layer: all accounting lives in
``portfolio_research_lab`` (the cash-deploy engine and the shared simulator used
for the baselines).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly (`streamlit run app/Home.py`) without installing the
# package, by adding the src/ layout to the import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from portfolio_research_lab import metrics  # noqa: E402
from portfolio_research_lab.cash_deploy import CashDeployResult, run_cash_deploy  # noqa: E402
from portfolio_research_lab.data import load_stocks_cash  # noqa: E402
from portfolio_research_lab.models import (  # noqa: E402
    PRESET_RULES,
    CashDeployConfig,
    DeployRule,
    StrategyConfig,
)
from portfolio_research_lab.simulator import run_simulation  # noqa: E402

SP500_DATA = PROJECT_ROOT / "data" / "sp500daily.csv"
FED_FUNDS_DATA = PROJECT_ROOT / "data" / "fed-funds-rate.csv"
STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"
CUSTOM_RULE = "Custom…"

st.set_page_config(page_title="Cash Deploy · Portfolio Research Lab", page_icon="💵", layout="wide")


@st.cache_data(show_spinner=False)
def _load_stocks_cash() -> pd.DataFrame:
    return load_stocks_cash(SP500_DATA, FED_FUNDS_DATA, stock_name=STOCK, cash_name=CASH)


def _metrics_row(label: str, m: dict[str, float], sharpe: float) -> dict[str, str]:
    return {
        "Strategy": label,
        "Total return": f"{m['total_return']:.2%}",
        "CAGR": f"{m['cagr']:.2%}",
        "Volatility (ann.)": f"{m['annualized_volatility']:.2%}",
        "Max drawdown": f"{m['max_drawdown']:.2%}",
        "Sharpe (vs cash)": f"{sharpe:.2f}",
    }


def _sharpe_vs_cash(equity: pd.Series, cash_level: pd.Series, periods_per_year: int) -> float:
    # Risk-free = the money-market leg's own periodic return, aligned to the
    # equity curve's return dates inside sharpe_ratio.
    returns = metrics.periodic_returns(equity)
    risk_free = cash_level.pct_change()
    return metrics.sharpe_ratio(returns, periods_per_year, risk_free)


def _sidebar_rule() -> DeployRule | None:
    """Collect a deploy rule from the sidebar (a preset or a custom one)."""
    choice = st.sidebar.selectbox("Deploy rule", [*PRESET_RULES, CUSTOM_RULE], index=3)
    if choice != CUSTOM_RULE:
        rule = PRESET_RULES[choice]
        th_str = " / ".join(f"{t:.0%}" for t in rule.thresholds)
        us_str = " / ".join(f"{u:.0%}" for u in rule.usages)
        st.sidebar.caption(f"Drawdown: {th_str}\n\nReserve used: {us_str}")
        return rule

    st.sidebar.caption(
        "One row per tranche: the drawdown that triggers it and the % of the "
        "reserve (locked at the drawdown's start) to deploy."
    )
    n = st.sidebar.number_input("Number of tranches", min_value=1, max_value=8, value=5)
    thresholds: list[float] = []
    usages: list[float] = []
    for i in range(int(n)):
        cols = st.sidebar.columns(2)
        th = cols[0].number_input(
            f"Drawdown {i + 1} (%)",
            min_value=1,
            max_value=99,
            value=min(10 * (i + 1), 99),
            key=f"th{i}",
        )
        us = cols[1].number_input(
            f"Reserve {i + 1} (%)", min_value=1, max_value=100, value=20, key=f"us{i}"
        )
        thresholds.append(th / 100.0)
        usages.append(us / 100.0)
    try:
        return DeployRule(name="Custom", thresholds=tuple(thresholds), usages=tuple(usages))
    except ValueError as exc:
        st.sidebar.error(f"Invalid rule: {exc}")
        return None


def _baseline_equity(
    prices: pd.DataFrame,
    allocations: dict[str, float],
    capital: float,
    *,
    rebalance: str | None,
    name: str,
) -> pd.Series:
    config = StrategyConfig.from_weights(
        allocations, name=name, initial_capital=capital, rebalance_frequency=rebalance
    )
    return run_simulation(prices, config).equity


def _equity_chart(result: CashDeployResult, baselines: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for label, series in baselines.items():
        fig.add_trace(
            go.Scatter(x=series.index, y=series, name=label, mode="lines", line={"width": 1})
        )
    fig.add_trace(
        go.Scatter(x=result.equity.index, y=result.equity, name=result.config.name, mode="lines")
    )
    fig.update_layout(
        margin={"t": 20, "b": 20}, yaxis_title="Portfolio value", hovermode="x unified"
    )
    return fig


def _composition_chart(result: CashDeployResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.stock_value.index,
            y=result.stock_value,
            name="Stocks",
            mode="lines",
            stackgroup="alloc",
            line={"width": 0.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.cash.index,
            y=result.cash,
            name="Cash reserve",
            mode="lines",
            stackgroup="alloc",
            line={"width": 0.5},
        )
    )
    fig.update_layout(margin={"t": 20, "b": 20}, yaxis_title="Value", hovermode="x unified")
    return fig


def _drawdown_chart(dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", mode="lines", fill="tozeroy"))
    fig.update_layout(
        margin={"t": 20, "b": 20},
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    st.title("💵 Cash Deploy")
    st.caption(
        "Hold a cash reserve, deploy it into stocks as the market falls, and "
        "refill it as the market recovers. For research and education only — "
        "not financial advice."
    )
    st.info(
        "**S&P 500 price return only** — dividends are not reinvested, which "
        "understates stock returns by roughly the dividend yield and mildly "
        "favours holding cash. Read the comparisons with that in mind."
    )

    try:
        prices = _load_stocks_cash()
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load price data: {exc}")
        st.stop()

    # --- Reserve & capital ------------------------------------------------
    st.sidebar.header("1 · Reserve")
    reserve_pct = st.sidebar.slider("Cash reserve (%)", 0, 100, 30, step=5) / 100.0
    st.sidebar.caption(f"Start: {1 - reserve_pct:.0%} stocks · {reserve_pct:.0%} cash")
    initial_capital = st.sidebar.number_input(
        "Initial capital", min_value=100.0, value=10_000.0, step=1_000.0
    )

    # --- Deploy rule ------------------------------------------------------
    st.sidebar.header("2 · Deploy rule")
    rule = _sidebar_rule()
    if rule is None:
        st.warning("Fix the deploy rule in the sidebar to run the backtest.")
        st.stop()
    if abs(rule.usage_sum - 1.0) > 1e-9:
        tail = (
            "Some reserve is never deployed."
            if rule.usage_sum < 1
            else "Deployment is capped by available cash."
        )
        st.sidebar.warning(f"Reserve usages sum to {rule.usage_sum:.0%}, not 100%. {tail}")

    # --- Refill & window --------------------------------------------------
    st.sidebar.header("3 · Refill")
    refill_rate = st.sidebar.slider("Refill rate (% of target / year)", 0, 100, 25, step=5) / 100.0
    st.sidebar.caption("Applied only while at a new all-time high.")

    st.sidebar.header("4 · Window")
    # `.map` sidesteps the missing DatetimeIndex.year type stub while staying a
    # plain int Index we can slice on.
    years = prices.index.map(lambda ts: ts.year)
    start_year = st.sidebar.slider(
        "Start year", int(years.min()), int(years.max()), int(years.min())
    )
    prices = prices[years >= start_year]
    if len(prices) < 2:
        st.warning("Not enough data in the selected window.")
        st.stop()

    st.sidebar.header("5 · Compare")
    show_stocks = st.sidebar.checkbox("100% stocks", value=True)
    show_static = st.sidebar.checkbox(
        f"Static {1 - reserve_pct:.0%}/{reserve_pct:.0%} (annual)", value=True
    )

    # --- Run --------------------------------------------------------------
    try:
        config = CashDeployConfig(
            name="Cash Deploy",
            initial_capital=initial_capital,
            reserve_pct=reserve_pct,
            rule=rule,
            refill_rate_per_year=refill_rate,
            stock_symbol=STOCK,
            cash_symbol=CASH,
        )
        result = run_cash_deploy(prices, config)
    except ValueError as exc:
        st.error(f"Simulation error: {exc}")
        st.stop()

    periods_per_year = config.trading_days_per_year
    cash_level = prices[CASH]

    baselines: dict[str, pd.Series] = {}
    if show_stocks:
        baselines["100% stocks"] = _baseline_equity(
            prices, {STOCK: 1.0}, initial_capital, rebalance=None, name="100% stocks"
        )
    if show_static and 0.0 < reserve_pct < 1.0:
        baselines[f"Static {1 - reserve_pct:.0%}/{reserve_pct:.0%}"] = _baseline_equity(
            prices,
            {STOCK: 1 - reserve_pct, CASH: reserve_pct},
            initial_capital,
            rebalance="annually",
            name="Static split",
        )

    # --- Metrics ----------------------------------------------------------
    st.subheader("Results")
    rows = [
        _metrics_row(
            config.name,
            result.metrics(),
            _sharpe_vs_cash(result.equity, cash_level, periods_per_year),
        )
    ]
    for label, series in baselines.items():
        rows.append(
            _metrics_row(
                label,
                metrics.summarize(series, periods_per_year),
                _sharpe_vs_cash(series, cash_level, periods_per_year),
            )
        )
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width="stretch")

    st.subheader("Equity curve")
    st.plotly_chart(_equity_chart(result, baselines), width="stretch")

    st.subheader("Reserve deployment")
    st.caption("Stacked value of the stock leg and the cash reserve over time.")
    st.plotly_chart(_composition_chart(result), width="stretch")

    st.subheader("Drawdown")
    st.plotly_chart(_drawdown_chart(result.drawdown()), width="stretch")

    with st.expander(f"Deploy / refill events ({len(result.events)})"):
        if result.events.empty:
            st.write("No deployments or refills in this window.")
        else:
            display = result.events.copy()
            display["drawdown"] = display["drawdown"].map(lambda x: f"{x:.1%}")
            display["amount"] = display["amount"].map(lambda x: f"{x:,.0f}")
            display["cash_after"] = display["cash_after"].map(lambda x: f"{x:,.0f}")
            st.dataframe(display, width="stretch")


if __name__ == "__main__":
    main()

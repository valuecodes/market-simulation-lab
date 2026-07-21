"""Trend Timing — a moving-average crossover backtest page.

Holds 100% stocks while the S&P is above its moving average and moves 100% to
cash (money-market at the fed funds rate) when it falls below, re-entering when
price climbs back above. Compares the result against a 100%-stocks buy-and-hold
and a static 60/40 split so the user can see whether trend timing actually helped
— the classic "does market timing beat buy-and-hold?" question.

Like ``1_Cash_Deploy.py`` this is a thin presentation layer: all accounting lives
in ``portfolio_research_lab`` (the timing engine and the shared simulator used for
the baselines).
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
from portfolio_research_lab.data import load_stocks_cash  # noqa: E402
from portfolio_research_lab.models import StrategyConfig, TimingConfig  # noqa: E402
from portfolio_research_lab.simulator import run_simulation  # noqa: E402
from portfolio_research_lab.timing import TimingResult, run_ma_timing  # noqa: E402

SP500_DATA = PROJECT_ROOT / "data" / "sp500daily.csv"
FED_FUNDS_DATA = PROJECT_ROOT / "data" / "fed-funds-rate.csv"
STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"
MA_KIND_LABELS: dict[str, str] = {"Simple (SMA)": "simple", "Exponential (EMA)": "exponential"}

st.set_page_config(
    page_title="Trend Timing · Portfolio Research Lab", page_icon="📈", layout="wide"
)


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


def _equity_chart(result: TimingResult, baselines: dict[str, pd.Series]) -> go.Figure:
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


def _signal_chart(prices: pd.DataFrame, result: TimingResult) -> go.Figure:
    """Stock price and its moving average, with the switch points marked."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices[STOCK], name="S&P 500", mode="lines", line={"width": 1})
    )
    fig.add_trace(
        go.Scatter(
            x=result.moving_average.index,
            y=result.moving_average,
            name="Moving average",
            mode="lines",
            line={"width": 1, "dash": "dot"},
        )
    )
    events = result.events
    for kind, symbol, color in (
        ("to_stocks", "triangle-up", "green"),
        ("to_cash", "triangle-down", "red"),
    ):
        hits = events[events["type"] == kind]
        if not hits.empty:
            fig.add_trace(
                go.Scatter(
                    x=hits.index,
                    y=hits["price"],
                    name="Enter stocks" if kind == "to_stocks" else "Exit to cash",
                    mode="markers",
                    marker={"symbol": symbol, "size": 9, "color": color},
                )
            )
    fig.update_layout(margin={"t": 20, "b": 20}, yaxis_title="Price", hovermode="x unified")
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
    st.title("📈 Trend Timing")
    st.caption(
        "Hold stocks while the price is above its moving average, step aside to "
        "cash when it falls below. For research and education only — not financial "
        "advice."
    )
    st.info(
        "**S&P 500 price return only** — dividends are not reinvested, which "
        "understates stock returns by roughly the dividend yield and mildly "
        "favours holding cash. Read the comparisons with that in mind."
    )
    st.warning(
        "**The signal is lagged one day (no look-ahead):** each day's position is "
        "decided from the *previous* close and traded at today's close. The "
        "200-day average is the conventional choice, not one fitted to this "
        "history — searching windows for the best backtest would be curve-fitting."
    )

    try:
        prices = _load_stocks_cash()
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load price data: {exc}")
        st.stop()

    # --- Signal -----------------------------------------------------------
    st.sidebar.header("1 · Signal")
    ma_window = st.sidebar.slider("Moving-average window (trading days)", 20, 300, 200, step=10)
    ma_kind_label = st.sidebar.selectbox("Average type", list(MA_KIND_LABELS), index=0)
    ma_kind = MA_KIND_LABELS[ma_kind_label]
    band_pct = st.sidebar.slider("Hysteresis band (%)", 0, 10, 0, step=1) / 100.0
    if band_pct > 0:
        st.sidebar.caption(
            f"Exit below -{band_pct:.0%} of the average, re-enter above +{band_pct:.0%}; "
            "hold inside the band."
        )

    # --- Costs & capital --------------------------------------------------
    st.sidebar.header("2 · Costs & capital")
    initial_capital = st.sidebar.number_input(
        "Initial capital", min_value=100.0, value=10_000.0, step=1_000.0
    )
    cost_bps = st.sidebar.slider("Transaction cost per switch (bps)", 0, 100, 0, step=5)
    st.sidebar.caption("10 bps = 0.10% of the traded amount, charged on each switch.")

    # --- Window -----------------------------------------------------------
    st.sidebar.header("3 · Window")
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
    st.sidebar.caption(
        "The average is recomputed on the selected window, so the strategy starts "
        "fully invested for the first "
        f"{ma_window - 1} trading days (warm-up)."
    )

    # --- Compare ----------------------------------------------------------
    st.sidebar.header("4 · Compare")
    show_stocks = st.sidebar.checkbox("100% stocks", value=True)
    show_static = st.sidebar.checkbox("Static 60/40 (annual)", value=True)

    # --- Run --------------------------------------------------------------
    try:
        config = TimingConfig(
            name="Trend Timing",
            initial_capital=initial_capital,
            ma_window=ma_window,
            ma_kind=ma_kind,
            band_pct=band_pct,
            cost_bps=float(cost_bps),
            stock_symbol=STOCK,
            cash_symbol=CASH,
        )
        result = run_ma_timing(prices, config)
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
    if show_static:
        baselines["Static 60/40"] = _baseline_equity(
            prices,
            {STOCK: 0.6, CASH: 0.4},
            initial_capital,
            rebalance="annually",
            name="Static 60/40",
        )

    # --- Headline ---------------------------------------------------------
    st.subheader("Results")
    c1, c2 = st.columns(2)
    c1.metric("Time in market", f"{result.time_in_market:.1%}")
    c2.metric("Switches", f"{result.n_switches:,}")

    rows = [
        _metrics_row(
            config.name,
            result.metrics(),
            metrics.sharpe_vs_cash(result.equity, cash_level, periods_per_year),
        )
    ]
    for label, series in baselines.items():
        rows.append(
            _metrics_row(
                label,
                metrics.summarize(series, periods_per_year),
                metrics.sharpe_vs_cash(series, cash_level, periods_per_year),
            )
        )
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width="stretch")

    st.subheader("Equity curve")
    st.plotly_chart(_equity_chart(result, baselines), width="stretch")

    st.subheader("Price vs moving average")
    st.caption(
        "Green ▲ = entered stocks, red ▼ = exited to cash (executed one day after the signal)."
    )
    st.plotly_chart(_signal_chart(prices, result), width="stretch")

    st.subheader("Drawdown")
    st.plotly_chart(_drawdown_chart(result.drawdown()), width="stretch")

    with st.expander(f"Switch events ({len(result.events)})"):
        if result.events.empty:
            st.write("No switches in this window — the strategy stayed fully invested.")
        else:
            display = result.events.copy()
            display["price"] = display["price"].map(lambda x: f"{x:,.2f}")
            display["ma"] = display["ma"].map(lambda x: f"{x:,.2f}")
            display["cost"] = display["cost"].map(lambda x: f"{x:,.2f}")
            display["value"] = display["value"].map(lambda x: f"{x:,.0f}")
            st.dataframe(display, width="stretch")


if __name__ == "__main__":
    main()

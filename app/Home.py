"""Portfolio Research Lab — Streamlit interface.

This module is the *only* place Streamlit is imported. It is a thin
presentation layer on top of the simulation engine in
``portfolio_research_lab``: it collects a configuration from the sidebar, calls
the engine, and renders the results. All calculations live in the engine.

Run it with::

    uv run streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly (`streamlit run app/Home.py`) without installing the
# package, by adding the src/ layout to the import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from portfolio_research_lab.data import (  # noqa: E402
    load_price_data,
    load_stocks_cash,
    parse_price_csv,
)
from portfolio_research_lab.models import StrategyConfig  # noqa: E402
from portfolio_research_lab.simulator import run_simulation  # noqa: E402

SP500_DATA = PROJECT_ROOT / "data" / "sp500daily.csv"
FED_FUNDS_DATA = PROJECT_ROOT / "data" / "fed-funds-rate.csv"

# Muted (low-opacity) colours for the per-asset 100%-allocation reference lines
# on the equity chart. Cycled if there are more assets than colours.
_REFERENCE_COLORS = (
    "rgba(99,110,250,0.40)",
    "rgba(239,85,59,0.40)",
    "rgba(0,204,150,0.40)",
    "rgba(171,99,250,0.40)",
)

st.set_page_config(page_title="Portfolio Research Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _load_sp500() -> pd.DataFrame:
    # Daily S&P 500 closing prices (data/sp500daily.csv, column `close`).
    # Rename to a readable asset name for the UI and metrics table.
    return load_price_data(SP500_DATA).rename(columns={"close": "S&P 500"})


@st.cache_data(show_spinner=False)
def _load_stocks_cash() -> pd.DataFrame:
    # Two-asset frame: S&P 500 alongside a synthetic cash account that compounds
    # the daily federal funds rate. Built by the shared engine loader
    # (data.load_stocks_cash), which the Cash Deploy page reuses too.
    return load_stocks_cash(SP500_DATA, FED_FUNDS_DATA)


def _load_uploaded(file) -> pd.DataFrame:
    # Parse the upload bytes directly through the engine's bounded, validating
    # parser (size / row / column / symbol-length limits + data-integrity checks).
    return parse_price_csv(file.getvalue())


def _metrics_row(label: str, m: dict[str, float]) -> dict[str, str]:
    return {
        "Strategy": label,
        "Total return": f"{m['total_return']:.2%}",
        "CAGR": f"{m['cagr']:.2%}",
        "Volatility (ann.)": f"{m['annualized_volatility']:.2%}",
        "Max drawdown": f"{m['max_drawdown']:.2%}",
    }


def main() -> None:
    st.title("📈 Portfolio Research Lab")
    st.caption(
        "A local-first lab for backtesting portfolio strategies. "
        "For research and education only — not financial advice."
    )

    # --- Data source ------------------------------------------------------
    st.sidebar.header("1 · Data")
    source = st.sidebar.radio(
        "Price data source",
        ["Stocks + Cash (1954+)", "S&P 500 (daily)", "Upload CSV"],
        index=0,
    )
    try:
        if source == "Stocks + Cash (1954+)":
            prices = _load_stocks_cash()
            st.sidebar.caption(
                "'Cash (Fed Funds)' compounds the daily federal funds rate — a "
                "risk-free short-rate proxy, without the duration/price risk of real bonds."
            )
        elif source == "S&P 500 (daily)":
            prices = _load_sp500()
        else:
            upload = st.sidebar.file_uploader("Wide-format CSV (date + asset columns)", type="csv")
            if upload is None:
                st.info("Upload a CSV, or switch to a bundled dataset, to begin.")
                st.stop()
            prices = _load_uploaded(upload)
    except (
        ValueError,  # also covers UnicodeDecodeError and pandas ParserError/EmptyDataError
        FileNotFoundError,
        KeyError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Could not load price data: {exc}")
        st.stop()

    assets = list(prices.columns)
    st.sidebar.success(
        f"Loaded {len(prices)} rows · {prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d}"
    )

    # --- Strategy configuration ------------------------------------------
    st.sidebar.header("2 · Strategy")
    name = st.sidebar.text_input("Strategy name", value="Buy & Hold")
    initial_capital = st.sidebar.number_input(
        "Initial capital", min_value=100.0, value=10_000.0, step=1_000.0
    )

    st.sidebar.subheader("Allocation weights (%)")
    if len(assets) == 2:
        # Two assets collapse to one intuitive slider: the left end is 100% of
        # the second asset, the right end 100% of the first. For the bundled
        # Stocks + Cash frame (columns ordered S&P 500, Cash) that reads as
        # "100% cash ↔ 100% stocks".
        right_asset, left_asset = assets[0], assets[1]
        st.sidebar.caption(f"0% → 100% {left_asset} · 100% → 100% {right_asset}")
        right_pct = st.sidebar.slider(f"% in {right_asset}", 0, 100, 60, step=5)
        raw_weights = {right_asset: right_pct, left_asset: 100 - right_pct}
    else:
        st.sidebar.caption("Assets are held passively; weights are normalised to 100%.")
        default_pct = round(100 / len(assets))
        raw_weights = {
            asset: st.sidebar.slider(asset, 0, 100, default_pct, step=5) for asset in assets
        }

    selected = {a: w for a, w in raw_weights.items() if w > 0}
    if not selected:
        st.warning("Give at least one asset a non-zero weight.")
        st.stop()

    rebalance_labels = {
        "None (buy & hold)": None,
        "Monthly": "monthly",
        "Quarterly": "quarterly",
        "Annually": "annually",
    }
    rebalance_choice = st.sidebar.selectbox("Rebalancing", list(rebalance_labels), index=0)
    rebalance_frequency = rebalance_labels[rebalance_choice]

    st.sidebar.header("3 · Benchmark")
    benchmark = st.sidebar.selectbox("Benchmark asset", ["(none)", *assets])
    benchmark_symbol = None if benchmark == "(none)" else benchmark

    # --- Run --------------------------------------------------------------
    try:
        config = StrategyConfig.from_weights(
            selected,
            normalize=True,
            name=name,
            initial_capital=initial_capital,
            benchmark=benchmark_symbol,
            rebalance_frequency=rebalance_frequency,
        )
        result = run_simulation(prices, config)
    except ValueError as exc:
        st.error(f"Simulation error: {exc}")
        st.stop()

    normalized = ", ".join(f"{a} {w:.0%}" for a, w in config.allocations.items())
    rebalance_note = (
        "buy & hold" if rebalance_frequency is None else f"rebalanced {rebalance_choice.lower()}"
    )
    st.subheader("Portfolio")
    st.write(f"**{config.name}** — {normalized} · {rebalance_note}")

    # --- Metrics ----------------------------------------------------------
    rows = [_metrics_row(config.name, result.metrics())]
    bench = result.benchmark_metrics()
    if bench is not None:
        rows.append(_metrics_row(f"{benchmark_symbol} (benchmark)", bench))
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width="stretch")

    # --- Equity curve -----------------------------------------------------
    st.subheader("Equity curve")
    equity_fig = go.Figure()
    # Dim reference lines: 100% buy-and-hold of each single asset, scaled to the
    # same starting capital, drawn underneath the portfolio for comparison.
    for i, asset in enumerate(assets):
        reference = initial_capital * prices[asset] / prices[asset].iloc[0]
        equity_fig.add_trace(
            go.Scatter(
                x=reference.index,
                y=reference,
                name=f"100% {asset}",
                mode="lines",
                line={"color": _REFERENCE_COLORS[i % len(_REFERENCE_COLORS)], "width": 1},
            )
        )
    equity_fig.add_trace(
        go.Scatter(x=result.equity.index, y=result.equity, name=config.name, mode="lines")
    )
    if result.benchmark_equity is not None:
        equity_fig.add_trace(
            go.Scatter(
                x=result.benchmark_equity.index,
                y=result.benchmark_equity,
                name=f"{benchmark_symbol} (benchmark)",
                mode="lines",
                line={"dash": "dash"},
            )
        )
    equity_fig.update_layout(
        margin={"t": 20, "b": 20}, yaxis_title="Portfolio value", hovermode="x unified"
    )
    st.plotly_chart(equity_fig, width="stretch")

    # --- Drawdown ---------------------------------------------------------
    st.subheader("Drawdown")
    dd = result.drawdown()
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", mode="lines", fill="tozeroy"))
    dd_fig.update_layout(
        margin={"t": 20, "b": 20},
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    st.plotly_chart(dd_fig, width="stretch")

    with st.expander("Show holdings & asset values"):
        holdings_note = (
            "Units held (constant for buy-and-hold):"
            if rebalance_frequency is None
            else "Units held (reset on each rebalance) — first & last rows:"
        )
        st.write(holdings_note)
        st.dataframe(
            pd.concat([result.holdings.head(1), result.holdings.tail(1)]),
            width="stretch",
        )
        st.write("Asset market values over time:")
        st.line_chart(result.asset_values)


if __name__ == "__main__":
    main()

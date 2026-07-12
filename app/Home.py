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
from io import StringIO
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
    generate_synthetic_prices,
    load_price_data,
)
from portfolio_research_lab.models import StrategyConfig  # noqa: E402
from portfolio_research_lab.simulator import run_simulation  # noqa: E402

SAMPLE_DATA = PROJECT_ROOT / "sample_data" / "sample_prices.csv"

st.set_page_config(page_title="Portfolio Research Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _load_sample() -> pd.DataFrame:
    # The bundled CSV is generated (and git-ignored). Create it on first run so
    # the app works immediately on a fresh clone.
    if not SAMPLE_DATA.exists():
        SAMPLE_DATA.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_prices().to_csv(SAMPLE_DATA)
    return load_price_data(SAMPLE_DATA)


def _load_uploaded(file) -> pd.DataFrame:
    text = file.getvalue().decode("utf-8")
    return load_price_data_from_text(text)


def load_price_data_from_text(text: str) -> pd.DataFrame:
    # Reuse the loader's cleaning rules by writing to an in-memory buffer.
    from portfolio_research_lab.data import DATE_COLUMN

    frame = pd.read_csv(StringIO(text))
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"expected a {DATE_COLUMN!r} column")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    frame = frame.set_index(DATE_COLUMN).sort_index().astype(float).ffill().dropna(how="any")
    return frame


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
    source = st.sidebar.radio("Price data source", ["Sample data", "Upload CSV"], index=0)
    try:
        if source == "Sample data":
            prices = _load_sample()
        else:
            upload = st.sidebar.file_uploader("Wide-format CSV (date + asset columns)", type="csv")
            if upload is None:
                st.info("Upload a CSV, or switch to sample data, to begin.")
                st.stop()
            prices = _load_uploaded(upload)
    except (ValueError, FileNotFoundError) as exc:
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
    st.sidebar.caption("Assets are held passively; weights are normalised to 100%.")
    default_pct = round(100 / len(assets))
    raw_weights: dict[str, float] = {}
    for asset in assets:
        raw_weights[asset] = st.sidebar.slider(asset, 0, 100, default_pct, step=5)

    selected = {a: w for a, w in raw_weights.items() if w > 0}
    if not selected:
        st.warning("Give at least one asset a non-zero weight.")
        st.stop()

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
        )
        result = run_simulation(prices, config)
    except ValueError as exc:
        st.error(f"Simulation error: {exc}")
        st.stop()

    normalized = ", ".join(f"{a} {w:.0%}" for a, w in config.allocations.items())
    st.subheader("Portfolio")
    st.write(f"**{config.name}** — {normalized}")

    # --- Metrics ----------------------------------------------------------
    rows = [_metrics_row(config.name, result.metrics())]
    bench = result.benchmark_metrics()
    if bench is not None:
        rows.append(_metrics_row(f"{benchmark_symbol} (benchmark)", bench))
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    # --- Equity curve -----------------------------------------------------
    st.subheader("Equity curve")
    equity_fig = go.Figure()
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
    st.plotly_chart(equity_fig, use_container_width=True)

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
    st.plotly_chart(dd_fig, use_container_width=True)

    with st.expander("Show holdings & asset values"):
        st.write("Units held (constant for buy-and-hold):")
        st.dataframe(result.holdings.head(1), use_container_width=True)
        st.write("Asset market values over time:")
        st.line_chart(result.asset_values)


if __name__ == "__main__":
    main()

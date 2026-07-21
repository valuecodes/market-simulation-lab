"""Optimal Cash Reserve Over Time — how the best reserve % drifts through history.

For a fixed deploy rule and refill rate, this sweeps the cash-reserve target at a
series of expanding-window snapshots: at each date it uses only the price history
*up to that point* (no look-ahead) and records the reserve that maximized the
chosen objective. It answers "standing here in history, what reserve would have
looked best on the past so far?" — an **in-sample fit at each date**, not an
out-of-sample claim (that is the Optimizer page's walk-forward job).

Like the other pages it is a thin presentation layer; all accounting and search
logic live in ``portfolio_research_lab``.
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

from portfolio_research_lab import metrics, optimizer  # noqa: E402
from portfolio_research_lab.data import load_stocks_cash  # noqa: E402
from portfolio_research_lab.models import PRESET_RULES, CashDeployConfig  # noqa: E402
from portfolio_research_lab.optimizer import (  # noqa: E402
    ObjectiveKind,
    ReserveSweepResult,
    SearchSpace,
    index_equity,
)

SP500_DATA = PROJECT_ROOT / "data" / "sp500daily.csv"
FED_FUNDS_DATA = PROJECT_ROOT / "data" / "fed-funds-rate.csv"
STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"

_OBJECTIVE_LABELS: dict[str, ObjectiveKind] = {
    "Excess CAGR vs index": ObjectiveKind.EXCESS_CAGR,
    "Excess CAGR, drawdown-capped": ObjectiveKind.EXCESS_CAGR_DD_CAPPED,
    "Sharpe (vs cash)": ObjectiveKind.SHARPE_VS_CASH,
    "CAGR": ObjectiveKind.CAGR,
}

st.set_page_config(
    page_title="Reserve Over Time · Portfolio Research Lab", page_icon="📈", layout="wide"
)


@st.cache_data(show_spinner=False)
def _load_stocks_cash() -> pd.DataFrame:
    return load_stocks_cash(SP500_DATA, FED_FUNDS_DATA, stock_name=STOCK, cash_name=CASH)


def _reserve_grid(steps: int) -> tuple[float, ...]:
    """Evenly spaced reserve grid over the optimizer's reserve range."""
    lo, hi = SearchSpace().reserve_range
    if steps < 2:
        return (lo,)
    return tuple(lo + (hi - lo) * i / (steps - 1) for i in range(steps))


def _reserve_line_chart(result: ReserveSweepResult, drawdown: pd.Series) -> go.Figure:
    """Optimal reserve over time, with the index drawdown on a secondary axis."""
    dates = [p.as_of for p in result.points]
    reserves = [p.optimal_reserve for p in result.points]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="Index drawdown",
            mode="lines",
            line={"width": 1, "color": "rgba(200,80,80,0.45)"},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=reserves,
            name="Optimal reserve",
            mode="lines+markers",
            line={"width": 2},
        )
    )
    fig.update_layout(
        margin={"t": 20, "b": 20},
        yaxis={"title": "Optimal cash reserve", "tickformat": ".0%"},
        yaxis2={
            "title": "Index drawdown",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "showgrid": False,
        },
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


def _objective_heatmap(result: ReserveSweepResult) -> go.Figure:
    """The full objective surface: reserve (y) by snapshot date (x), coloured by objective."""
    dates = [p.as_of for p in result.points]
    # Numeric reserves for the y-axis (formatted via tickformat): using rounded
    # percent strings would collapse distinct grid values into one row at higher
    # reserve resolutions.
    reserves = list(result.reserve_grid)
    # z[i][j] = objective at reserve i, snapshot j (columns aligned with dates).
    z = [
        [p.objective_by_reserve[i] for p in result.points] for i in range(len(result.reserve_grid))
    ]
    fig = go.Figure(
        go.Heatmap(
            x=dates,
            y=reserves,
            z=z,
            colorscale="Viridis",
            colorbar={"title": "Objective"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[p.optimal_reserve for p in result.points],
            name="Optimal",
            mode="lines+markers",
            line={"color": "white", "width": 2},
            marker={"size": 5, "color": "white"},
        )
    )
    fig.update_layout(
        margin={"t": 20, "b": 20},
        xaxis_title="Snapshot date",
        yaxis={"title": "Cash reserve", "tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


def _snapshot_table(result: ReserveSweepResult) -> pd.DataFrame:
    rows = [
        {
            "As of": f"{p.as_of:%Y-%m-%d}",
            "History (rows)": p.n_rows,
            "Optimal reserve": f"{p.optimal_reserve:.1%}",
            "In-sample excess CAGR": f"{p.excess_cagr:+.2%}",
        }
        for p in result.points
    ]
    return pd.DataFrame(rows).set_index("As of")


def main() -> None:
    st.title("📈 Optimal Cash Reserve Over Time")
    st.caption(
        "For a fixed deploy rule and refill rate, sweep the cash-reserve target at a "
        "series of expanding-window snapshots and track the reserve that looked best "
        "on the history up to each date. Research and education only — not financial advice."
    )
    st.warning(
        "**Each point is an in-sample fit.** At every date the reserve is the one that "
        "maximized the objective *on the past so far* — no future data is used (no "
        "look-ahead), but this is still curve-fitting to history, not an out-of-sample "
        "result. For honest out-of-sample numbers use the **Optimizer** page's walk-forward."
    )
    st.info(
        "**S&P 500 price return only** — dividends are not reinvested, which understates "
        "stock returns and mildly favours holding cash, so a higher 'optimal' reserve here "
        "is partly an artefact of the price-return data."
    )

    try:
        prices = _load_stocks_cash()
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load price data: {exc}")
        st.stop()

    # --- Controls ---------------------------------------------------------
    st.sidebar.header("1 · Objective")
    objective_label = st.sidebar.selectbox("Maximize", list(_OBJECTIVE_LABELS), index=0)
    objective_kind = _OBJECTIVE_LABELS[objective_label]
    dd_cap = 0.35
    if objective_kind is ObjectiveKind.EXCESS_CAGR_DD_CAPPED:
        dd_cap = st.sidebar.slider("Max drawdown cap (%)", 10, 80, 35, step=5) / 100.0

    st.sidebar.header("2 · Fixed strategy")
    st.sidebar.caption("The deploy rule and refill are held fixed; only the reserve is swept.")
    rule_name = st.sidebar.selectbox(
        "Deploy rule", list(PRESET_RULES), index=list(PRESET_RULES).index("Recommended")
    )
    rule = PRESET_RULES[rule_name]
    refill = st.sidebar.slider("Refill rate (% / year)", 0, 100, 25, step=5) / 100.0

    st.sidebar.header("3 · Grid & window")
    n_points = st.sidebar.slider("Time snapshots", 4, 40, 24)
    reserve_steps = st.sidebar.slider("Reserve resolution", 5, 41, 21, step=2)
    warmup_years = st.sidebar.slider("Warm-up (years)", 1, 15, 5)

    base = CashDeployConfig(
        name="Reserve sweep",
        rule=rule,
        stock_symbol=STOCK,
        cash_symbol=CASH,
    )
    min_rows = warmup_years * base.trading_days_per_year

    # `.map` rather than the vectorized `.index.year`: ty types `.index` as the
    # generic Index, which has no `.year`; this mirrors 2_Optimize.py.
    years = prices.index.map(lambda ts: ts.year)
    start_year = st.sidebar.slider(
        "Start year", int(years.min()), int(years.max()), int(years.min())
    )
    prices = prices[years >= start_year]

    grid = _reserve_grid(reserve_steps)
    total_backtests = n_points * len(grid)
    st.sidebar.caption(
        f"≈ {total_backtests:,} backtests "
        f"(~{total_backtests * 0.04:.0f}s at ~40ms each). Deterministic — no random seed."
    )

    run = st.sidebar.button("Run sweep", type="primary")

    if run:
        progress = st.progress(0.0)
        status = st.empty()
        last_update = {"n": 0}

        def on_progress(done: int, total: int, snapshot_best: float) -> None:
            if done - last_update["n"] >= len(grid) or done >= total:
                last_update["n"] = done
                progress.progress(min(done / total, 1.0))
                snapshot = -(-done // len(grid))  # ceil division → 1-based snapshot index
                status.write(
                    f"Snapshot {snapshot:,}/{n_points:,} · "
                    f"best objective in this snapshot: {snapshot_best:.4f}"
                )

        try:
            with st.spinner("Sweeping…"):
                result = optimizer.optimal_reserve_over_time(
                    prices,
                    base=base,
                    rule=rule,
                    refill_rate_per_year=refill,
                    objective_kind=objective_kind,
                    reserve_grid=grid,
                    n_points=n_points,
                    min_rows=min_rows,
                    dd_cap=dd_cap,
                    on_progress=on_progress,
                )
        except ValueError as exc:
            progress.empty()
            status.empty()
            st.warning(f"Could not run the sweep: {exc}")
            st.stop()

        progress.progress(1.0)
        status.write(f"Done — evaluated {total_backtests:,} backtests.")
        st.session_state["reserve_result"] = result
        st.session_state["reserve_prices"] = prices

    result = st.session_state.get("reserve_result")
    if result is None:
        st.info("Set the controls in the sidebar and press **Run sweep**.")
        return

    prices = st.session_state["reserve_prices"]
    latest = result.points[-1]

    # --- Headline ---------------------------------------------------------
    st.subheader("Latest snapshot")
    c1, c2 = st.columns(2)
    c1.metric(f"Optimal reserve — as of {latest.as_of:%Y-%m-%d}", f"{latest.optimal_reserve:.1%}")
    c2.metric("In-sample excess CAGR vs index", f"{latest.excess_cagr:+.2%}")
    st.caption(
        f"Fitted on {latest.n_rows:,} rows of history through "
        f"**{latest.as_of:%Y-%m-%d}** — the last observation in the data, not "
        f"necessarily the present day. Deploy rule: **{result.rule.name}**, "
        f"refill **{result.refill_rate_per_year:.0%}/yr**."
    )

    # --- Trajectory -------------------------------------------------------
    st.subheader("Optimal reserve over time")
    st.caption(
        "How the best in-sample reserve drifts as history accumulates, against the "
        "index drawdown for context — does dry powder look better after crashes?"
    )
    idx_drawdown = metrics.drawdown_series(index_equity(prices, base))
    st.plotly_chart(_reserve_line_chart(result, idx_drawdown), width="stretch")

    # --- Objective surface ------------------------------------------------
    st.subheader("Objective surface")
    st.caption(
        "The objective at every reserve and snapshot. A flat column means the reserve "
        "barely mattered at that date; a sharp ridge means it mattered a lot."
    )
    st.plotly_chart(_objective_heatmap(result), width="stretch")

    st.subheader("Snapshots")
    st.dataframe(_snapshot_table(result), width="stretch")


if __name__ == "__main__":
    main()

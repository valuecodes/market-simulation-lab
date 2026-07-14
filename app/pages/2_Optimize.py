"""Strategy Optimizer — search cash-deploy parameters that beat the index.

Runs a Bayesian (Optuna TPE) parameter search over the cash-deploy strategy's
cash reserve, refill rate and drawdown-triggered deploy tranches, looking for
configurations that beat a 100%-stocks buy-and-hold. Honesty about
generalization comes from walk-forward validation: parameters are fitted on a
train window and scored on a later, unseen test window, and the out-of-sample
result is what the page leads with.

This is *parameter search over history*, not machine learning — see the warning
on the page. Like ``1_Cash_Deploy.py`` it is a thin presentation layer; all
accounting and search logic live in ``portfolio_research_lab``.
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
from portfolio_research_lab.cash_deploy import run_cash_deploy  # noqa: E402
from portfolio_research_lab.data import load_stocks_cash  # noqa: E402
from portfolio_research_lab.models import PRESET_RULES, CashDeployConfig  # noqa: E402
from portfolio_research_lab.optimizer import (  # noqa: E402
    ObjectiveKind,
    SearchSpace,
    WalkForwardResult,
    index_equity,
)

SP500_DATA = PROJECT_ROOT / "data" / "sp500daily.csv"
FED_FUNDS_DATA = PROJECT_ROOT / "data" / "fed-funds-rate.csv"
STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"

# Reserve/refill applied to the preset rules in the comparison table (the
# page-1 defaults), so the presets are judged on their rule, not tuned settings.
PRESET_RESERVE = 0.30
PRESET_REFILL = 0.25

_OBJECTIVE_LABELS: dict[str, ObjectiveKind] = {
    "Excess CAGR vs index": ObjectiveKind.EXCESS_CAGR,
    "Excess CAGR, drawdown-capped": ObjectiveKind.EXCESS_CAGR_DD_CAPPED,
    "Sharpe (vs cash)": ObjectiveKind.SHARPE_VS_CASH,
    "CAGR": ObjectiveKind.CAGR,
}

st.set_page_config(page_title="Optimizer · Portfolio Research Lab", page_icon="🔎", layout="wide")


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


def _comparison_table(prices: pd.DataFrame, best: CashDeployConfig) -> pd.DataFrame:
    """Optimized config vs 100% stocks and the four presets over the full window."""
    ppy = best.trading_days_per_year
    cash_level = prices[CASH]
    rows: list[dict[str, str]] = []

    result = run_cash_deploy(prices, best)
    rows.append(
        _metrics_row(
            "Optimized",
            result.metrics(),
            metrics.sharpe_vs_cash(result.equity, cash_level, ppy),
        )
    )

    index_eq = index_equity(prices, best)
    rows.append(
        _metrics_row(
            "100% stocks (index)",
            metrics.summarize(index_eq, ppy),
            metrics.sharpe_vs_cash(index_eq, cash_level, ppy),
        )
    )

    for name, rule in PRESET_RULES.items():
        cfg = best.model_copy(
            update={
                "reserve_pct": PRESET_RESERVE,
                "refill_rate_per_year": PRESET_REFILL,
                "rule": rule,
            }
        )
        preset_result = run_cash_deploy(prices, cfg)
        rows.append(
            _metrics_row(
                f"Preset · {name}",
                preset_result.metrics(),
                metrics.sharpe_vs_cash(preset_result.equity, cash_level, ppy),
            )
        )

    return pd.DataFrame(rows).set_index("Strategy")


def _bucket_table(best: CashDeployConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Drawdown trigger": [f"{t:.1%}" for t in best.rule.thresholds],
            "Reserve deployed": [f"{u:.1%}" for u in best.rule.usages],
        }
    )


def _walk_forward_table(wf: WalkForwardResult) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(wf.folds, start=1):
        rows.append(
            {
                "Fold": i,
                "Train": f"{fold.train_window[0]:%Y-%m} → {fold.train_window[1]:%Y-%m}",
                "Test": f"{fold.test_window[0]:%Y-%m} → {fold.test_window[1]:%Y-%m}",
                "Train excess CAGR": f"{fold.train_excess_cagr:+.2%}",
                "Test excess CAGR": f"{fold.test_excess_cagr:+.2%}",
            }
        )
    return pd.DataFrame(rows).set_index("Fold")


def _equity_chart(strategy: pd.Series, index: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=index.index, y=index, name="100% stocks", mode="lines", line={"width": 1})
    )
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy, name="Optimized", mode="lines"))
    fig.update_layout(
        margin={"t": 20, "b": 20}, yaxis_title="Portfolio value", hovermode="x unified"
    )
    return fig


def _history_chart(history: pd.DataFrame) -> go.Figure:
    running_best = history["value"].cummax()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["trial"],
            y=history["value"],
            name="Trial",
            mode="markers",
            marker={"size": 4, "opacity": 0.5},
        )
    )
    fig.add_trace(go.Scatter(x=history["trial"], y=running_best, name="Best so far", mode="lines"))
    fig.update_layout(
        margin={"t": 20, "b": 20},
        xaxis_title="Trial",
        yaxis_title="Objective",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    st.title("🔎 Strategy Optimizer")
    st.caption(
        "Search the cash-deploy parameters — cash reserve, refill rate and "
        "drawdown deploy tranches — for settings that beat the index. For "
        "research and education only — not financial advice."
    )
    st.warning(
        "**This fits parameters to the past.** Picking the best of many "
        "configurations on one historical price path is curve-fitting: a config "
        "that beat the index historically will not necessarily do so in future. "
        "Trust the **out-of-sample** (test) column, not the in-sample one."
    )
    st.info(
        "**S&P 500 price return only** — dividends are not reinvested, which "
        "understates stock returns and mildly favours holding cash, so the bar "
        "to 'beat the index' here is easier than against a total-return index."
    )

    try:
        prices = _load_stocks_cash()
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load price data: {exc}")
        st.stop()

    # --- Search controls --------------------------------------------------
    st.sidebar.header("1 · Objective")
    objective_label = st.sidebar.selectbox("Maximize", list(_OBJECTIVE_LABELS), index=0)
    objective_kind = _OBJECTIVE_LABELS[objective_label]
    dd_cap = 0.35
    if objective_kind is ObjectiveKind.EXCESS_CAGR_DD_CAPPED:
        dd_cap = st.sidebar.slider("Max drawdown cap (%)", 10, 80, 35, step=5) / 100.0

    st.sidebar.header("2 · Search space")
    max_buckets = st.sidebar.slider("Max deploy buckets", 1, 5, 5)

    st.sidebar.header("3 · Budget & validation")
    n_trials = st.sidebar.slider("Trials per fold", 25, 1000, 200, step=25)
    n_folds = st.sidebar.slider("Walk-forward folds", 2, 8, 5)
    train_frac = st.sidebar.slider("Train fraction per fold", 0.3, 0.8, 0.5, step=0.05)
    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=0, step=1))

    st.sidebar.header("4 · Window")
    years = prices.index.map(lambda ts: ts.year)
    start_year = st.sidebar.slider(
        "Start year", int(years.min()), int(years.max()), int(years.min())
    )
    prices = prices[years >= start_year]
    if len(prices) < 100:
        st.warning("Not enough data in the selected window to optimize.")
        st.stop()

    total_trials = (n_folds + 1) * n_trials
    st.sidebar.caption(f"≈ {total_trials:,} backtests (~{total_trials * 0.04:.0f}s at ~40ms each).")

    run = st.sidebar.button("Run search", type="primary")

    if run:
        space = SearchSpace(max_buckets=max_buckets)
        base = CashDeployConfig(
            name="Optimized",
            rule=PRESET_RULES["Recommended"],  # placeholder; overwritten by the search
            stock_symbol=STOCK,
            cash_symbol=CASH,
        )
        progress = st.progress(0.0)
        status = st.empty()
        last_update = {"n": 0}

        def on_trial(done: int, total: int, best: float) -> None:
            if done - last_update["n"] >= 25 or done >= total:
                last_update["n"] = done
                progress.progress(min(done / total, 1.0))
                status.write(f"Trial {done:,}/{total:,} · best objective so far: {best:.4f}")

        with st.spinner("Searching…"):
            wf = optimizer.walk_forward(
                prices,
                base=base,
                objective_kind=objective_kind,
                space=space,
                n_folds=n_folds,
                train_frac=train_frac,
                n_trials=n_trials,
                seed=seed,
                dd_cap=dd_cap,
            )
        progress.progress(1.0)
        status.write(f"Done — evaluated {total_trials:,} configurations.")
        st.session_state["opt_result"] = wf
        st.session_state["opt_prices"] = prices

    wf = st.session_state.get("opt_result")
    if wf is None:
        st.info("Set the search controls in the sidebar and press **Run search**.")
        return

    prices = st.session_state["opt_prices"]
    best = wf.final.best

    # --- Recommended config ----------------------------------------------
    st.subheader("Recommended configuration")
    st.caption("Optimized over the full selected window — the parameters you would deploy.")
    c1, c2 = st.columns(2)
    c1.metric("Cash reserve target", f"{best.reserve_pct:.1%}")
    c2.metric("Refill rate", f"{best.refill_rate_per_year:.1%} / year")
    st.table(_bucket_table(best))

    # --- Out-of-sample validation ----------------------------------------
    st.subheader("Walk-forward validation (out-of-sample)")
    st.caption(
        "Each fold optimizes on its train window, then measures the frozen winner "
        "on the later, unseen test window. The **test** column is the honest number."
    )
    st.metric("Mean out-of-sample excess CAGR vs index", f"{wf.mean_test_excess_cagr:+.2%}")
    st.dataframe(_walk_forward_table(wf), width="stretch")
    st.caption(
        f"Beat the index **in-sample** in {wf.n_beat_index_train}/{len(wf.folds)} folds; "
        f"**out-of-sample** in {wf.n_beat_index_test}/{len(wf.folds)} folds. "
        "A large gap between the two is the curve-fitting tax."
    )

    # --- Comparison -------------------------------------------------------
    st.subheader("Comparison over the full window")
    st.caption(
        f"Presets use the default {PRESET_RESERVE:.0%} reserve / {PRESET_REFILL:.0%} refill, "
        "so they are judged on their deploy rule."
    )
    st.dataframe(_comparison_table(prices, best), width="stretch")

    st.subheader("Equity curve")
    result = run_cash_deploy(prices, best)
    st.plotly_chart(_equity_chart(result.equity, index_equity(prices, best)), width="stretch")

    st.subheader("Search progress (final fit)")
    st.caption("Objective value per trial and the running best, over the full-window fit.")
    st.plotly_chart(_history_chart(wf.final.history), width="stretch")


if __name__ == "__main__":
    main()

"""Strategy optimizer for the cash-deploy backtest.

This module searches the cash-deploy parameter space — cash reserve, refill rate,
and the drawdown-triggered deploy tranches — for configurations that *beat the
index* (100%-stocks buy-and-hold) over historical price data.

It is **parameter search over a single historical price path, not supervised
machine learning**: there is no labelled training set, only a different slice of
the same S&P series to validate against. The search uses Optuna's TPE (Bayesian)
sampler; honesty about generalization comes from :func:`walk_forward`, which
optimizes on a train window and measures the frozen winner on a later, unseen
test window.

Like the rest of :mod:`portfolio_research_lab`, nothing here imports Streamlit,
so the optimizer runs from notebooks, scripts and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import optuna
import pandas as pd

from portfolio_research_lab import metrics
from portfolio_research_lab.cash_deploy import CashDeployResult, run_cash_deploy
from portfolio_research_lab.models import CashDeployConfig, DeployRule, StrategyConfig
from portfolio_research_lab.simulator import run_simulation

# Sorted grid of candidate drawdown thresholds. Sampling a distinct subset makes
# every candidate's thresholds strictly-increasing-in-(0, 1) *by construction* —
# no repair loop, no rejected trials.
THRESHOLD_GRID: tuple[float, ...] = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# How many top configurations to keep for the leaderboard / overfitting diagnostic.
LEADERBOARD_SIZE = 20


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Bounds and grids the optimizer samples parameters from.

    Ranges are deliberately tighter than the model's full legal domain so the
    search spends its budget on sensible regions (e.g. a reserve of 5-60% rather
    than the model-legal 0-100%). Everything is overridable from the UI.
    """

    threshold_grid: tuple[float, ...] = THRESHOLD_GRID
    max_buckets: int = 5
    usage_range: tuple[float, float] = (0.05, 1.0)
    reserve_range: tuple[float, float] = (0.05, 0.60)
    refill_range: tuple[float, float] = (0.0, 1.0)


class ObjectiveKind(Enum):
    """What the optimizer maximizes. Higher is always better."""

    EXCESS_CAGR = "excess_cagr"
    EXCESS_CAGR_DD_CAPPED = "excess_cagr_dd_capped"
    SHARPE_VS_CASH = "sharpe_vs_cash"
    CAGR = "cagr"


@dataclass(frozen=True, slots=True)
class ObjectiveContext:
    """Everything an objective needs beyond the strategy result itself.

    ``benchmark_cagr`` is the 100%-stocks buy-and-hold CAGR over the *same* slice
    the strategy was run on — the "index" the strategy must beat. ``cash_level``
    is that slice's money-market leg, used as the risk-free rate for Sharpe.
    """

    benchmark_cagr: float
    cash_level: pd.Series
    periods_per_year: int
    dd_cap: float = 0.35
    dd_penalty: float = 1.0


# An objective scores a run from its result and its already-computed metrics
# (passed in so metrics() is not recomputed per trial).
Objective = Callable[[CashDeployResult, dict[str, float]], float]


def make_objective(kind: ObjectiveKind, ctx: ObjectiveContext) -> Objective:
    """Build the scoring function for ``kind``, closing over ``ctx``."""

    def excess_cagr(result: CashDeployResult, m: dict[str, float]) -> float:
        return m["cagr"] - ctx.benchmark_cagr

    def excess_cagr_dd_capped(result: CashDeployResult, m: dict[str, float]) -> float:
        excess = m["cagr"] - ctx.benchmark_cagr
        # max_drawdown is <= 0; -max_drawdown is the positive drawdown magnitude.
        breach = max(0.0, -m["max_drawdown"] - ctx.dd_cap)
        return excess - ctx.dd_penalty * breach

    def sharpe(result: CashDeployResult, m: dict[str, float]) -> float:
        return metrics.sharpe_vs_cash(result.equity, ctx.cash_level, ctx.periods_per_year)

    def cagr(result: CashDeployResult, m: dict[str, float]) -> float:
        return m["cagr"]

    dispatch: dict[ObjectiveKind, Objective] = {
        ObjectiveKind.EXCESS_CAGR: excess_cagr,
        ObjectiveKind.EXCESS_CAGR_DD_CAPPED: excess_cagr_dd_capped,
        ObjectiveKind.SHARPE_VS_CASH: sharpe,
        ObjectiveKind.CAGR: cagr,
    }
    return dispatch[kind]


def suggest_config(
    trial: optuna.Trial,
    space: SearchSpace,
    base: CashDeployConfig,
) -> CashDeployConfig:
    """Sample one candidate :class:`CashDeployConfig` from ``space``.

    ``base`` supplies the fixed context (symbols, capital, trading days, name);
    only ``reserve_pct``, ``refill_rate_per_year`` and the deploy ``rule`` are
    searched. The threshold encoding samples ``n`` grid indices and decodes the
    *distinct* ones sorted, so the resulting rule always satisfies
    :class:`DeployRule`'s constraints (a collision simply yields fewer tranches).
    """
    reserve_pct = trial.suggest_float("reserve_pct", *space.reserve_range)
    refill = trial.suggest_float("refill_rate_per_year", *space.refill_range)

    n = trial.suggest_int("n_buckets", 1, space.max_buckets)
    grid = space.threshold_grid
    indices = {trial.suggest_int(f"t{k}", 0, len(grid) - 1) for k in range(n)}
    thresholds = tuple(sorted(grid[i] for i in indices))
    usages = tuple(trial.suggest_float(f"u{k}", *space.usage_range) for k in range(len(thresholds)))

    rule = DeployRule(name="search", thresholds=thresholds, usages=usages)
    return base.model_copy(
        update={"reserve_pct": reserve_pct, "refill_rate_per_year": refill, "rule": rule}
    )


@dataclass(slots=True)
class ScoredConfig:
    """A candidate config with its objective score and headline metrics (in-sample)."""

    config: CashDeployConfig
    score: float
    metrics: dict[str, float]


@dataclass(slots=True)
class OptimizationResult:
    """Result of a single train/test optimization.

    ``test_*`` fields are ``None`` when there is no holdout (``split >= 1.0``).
    ``test_metrics`` — the *frozen* best config re-run as an independent
    simulation on the unseen test slice — is the honest, out-of-sample headline.
    """

    best: CashDeployConfig
    train_window: tuple[pd.Timestamp, pd.Timestamp]
    test_window: tuple[pd.Timestamp, pd.Timestamp] | None
    train_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    train_index_metrics: dict[str, float]
    test_index_metrics: dict[str, float] | None
    leaderboard: list[ScoredConfig]
    history: pd.DataFrame
    n_evaluated: int


@dataclass(slots=True)
class Fold:
    """One walk-forward fold: optimize on ``train_window``, measure on ``test_window``."""

    train_window: tuple[pd.Timestamp, pd.Timestamp]
    test_window: tuple[pd.Timestamp, pd.Timestamp]
    config: CashDeployConfig
    train_excess_cagr: float
    test_excess_cagr: float
    result: OptimizationResult


@dataclass(slots=True)
class WalkForwardResult:
    """Walk-forward validation plus the final full-history recommendation.

    ``mean_test_excess_cagr`` is the number to trust — the average out-of-sample
    margin over the index across folds. ``final`` is the config optimized over the
    entire window; it is what you would actually deploy.
    """

    folds: list[Fold]
    mean_test_excess_cagr: float
    final: OptimizationResult
    n_beat_index_train: int = field(init=False)
    n_beat_index_test: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_beat_index_train = sum(1 for f in self.folds if f.train_excess_cagr > 0)
        self.n_beat_index_test = sum(1 for f in self.folds if f.test_excess_cagr > 0)


ProgressCallback = Callable[[int, int, float], None]


def _index_metrics(
    prices: pd.DataFrame, base: CashDeployConfig
) -> tuple[dict[str, float], pd.Series]:
    """Headline metrics and equity curve of a 100%-stocks buy-and-hold over ``prices``."""
    config = StrategyConfig.from_weights(
        {base.stock_symbol: 1.0},
        name="100% stocks",
        initial_capital=base.initial_capital,
        trading_days_per_year=base.trading_days_per_year,
    )
    equity = run_simulation(prices, config).equity
    return metrics.summarize(equity, base.trading_days_per_year), equity


def index_equity(prices: pd.DataFrame, base: CashDeployConfig) -> pd.Series:
    """The 100%-stocks buy-and-hold equity curve over ``prices`` (for charts)."""
    return _index_metrics(prices, base)[1]


def _build_context(prices: pd.DataFrame, base: CashDeployConfig, dd_cap: float) -> ObjectiveContext:
    index_m, _ = _index_metrics(prices, base)
    return ObjectiveContext(
        benchmark_cagr=index_m["cagr"],
        cash_level=prices[base.cash_symbol],
        periods_per_year=base.trading_days_per_year,
        dd_cap=dd_cap,
    )


def optimize(
    prices: pd.DataFrame,
    *,
    base: CashDeployConfig,
    objective_kind: ObjectiveKind = ObjectiveKind.EXCESS_CAGR,
    space: SearchSpace | None = None,
    split: float = 0.60,
    n_trials: int = 200,
    seed: int = 0,
    dd_cap: float = 0.35,
    on_trial: ProgressCallback | None = None,
) -> OptimizationResult:
    """Search ``prices`` for the best cash-deploy config, honest about out-of-sample.

    The rows are split chronologically at ``split`` (a fraction in ``(0, 1]``).
    The Optuna study runs on the **train** slice only; the frozen best config is
    then re-run as an independent simulation on the **test** slice and reported
    separately. ``split >= 1.0`` disables the holdout (train = full window), used
    for the final deployable fit.

    ``on_trial(done, total, best_value)`` is called after each trial for progress.
    """
    if space is None:
        space = SearchSpace()
    if len(prices) < 2:
        raise ValueError("need at least two price rows to optimize")

    # Keep Optuna's per-trial INFO logging quiet; done here rather than at import
    # so `import portfolio_research_lab` has no global logging side effect.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cut = len(prices) if split >= 1.0 else round(len(prices) * split)
    cut = max(2, min(cut, len(prices)))
    train = prices.iloc[:cut]
    test = prices.iloc[cut:]
    has_test = len(test) >= 2
    if split < 1.0 and not has_test:
        # A split below 1.0 promises a holdout; refuse to silently drop it.
        raise ValueError(
            f"not enough rows ({len(prices)}) for a holdout at split={split}; "
            "use more data or split=1.0"
        )

    train_ctx = _build_context(train, base, dd_cap)
    objective = make_objective(objective_kind, train_ctx)

    scored: list[ScoredConfig] = []

    def objective_fn(trial: optuna.Trial) -> float:
        config = suggest_config(trial, space, base)
        result = run_cash_deploy(train, config)
        m = result.metrics()
        score = objective(result, m)
        scored.append(ScoredConfig(config=config, score=score, metrics=m))
        return score

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if on_trial is None:
            return
        # study.best_value raises until at least one trial has completed (e.g. if
        # early trials prune), so guard it — progress must never crash the search.
        try:
            best = study.best_value
        except ValueError:
            return
        on_trial(trial.number + 1, n_trials, best)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, callbacks=[callback])

    if not scored:
        raise RuntimeError("no configurations were successfully evaluated")

    leaderboard = sorted(scored, key=lambda s: s.score, reverse=True)[:LEADERBOARD_SIZE]
    best = leaderboard[0]

    train_index_m, _ = _index_metrics(train, base)
    test_metrics: dict[str, float] | None = None
    test_index_metrics: dict[str, float] | None = None
    test_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if has_test:
        # Re-run the frozen best as an INDEPENDENT simulation on the unseen slice:
        # the engine resets its peak/reserve to the slice's first row, so the test
        # result must never be sliced out of the train equity curve.
        test_metrics = run_cash_deploy(test, best.config).metrics()
        test_index_metrics, _ = _index_metrics(test, base)
        test_window = (test.index[0], test.index[-1])

    history = pd.DataFrame({"trial": range(len(scored)), "value": [s.score for s in scored]})

    return OptimizationResult(
        best=best.config,
        train_window=(train.index[0], train.index[-1]),
        test_window=test_window,
        train_metrics=best.metrics,
        test_metrics=test_metrics,
        train_index_metrics=train_index_m,
        test_index_metrics=test_index_metrics,
        leaderboard=leaderboard,
        history=history,
        n_evaluated=len(scored),
    )


def walk_forward(
    prices: pd.DataFrame,
    *,
    base: CashDeployConfig,
    objective_kind: ObjectiveKind = ObjectiveKind.EXCESS_CAGR,
    space: SearchSpace | None = None,
    n_folds: int = 5,
    train_frac: float = 0.5,
    n_trials: int = 200,
    seed: int = 0,
    dd_cap: float = 0.35,
    on_trial: ProgressCallback | None = None,
) -> WalkForwardResult:
    """Walk-forward validation: roll ``n_folds`` non-overlapping train->test windows.

    The timeline is cut into ``n_folds`` contiguous windows; each is split
    internally at ``train_frac`` into a train part (optimized on) and a later test
    part (measured on the frozen winner). This yields an honest out-of-sample
    excess-CAGR per fold with no leakage — each fold's test rows are strictly later
    than its own train rows, and folds do not overlap.

    ``final`` is one more optimization over the *entire* window (no holdout): the
    single recommended parameter set to deploy. Progress is reported across all
    ``(n_folds + 1) * n_trials`` trials.
    """
    if space is None:
        space = SearchSpace()
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")

    n = len(prices)
    window = n // n_folds
    if window < 4:
        raise ValueError(f"not enough data for {n_folds} folds over {n} rows")

    total_trials = (n_folds + 1) * n_trials
    completed = 0

    def fold_progress(done: int, _total: int, best: float) -> None:
        if on_trial is not None:
            on_trial(completed + done, total_trials, best)

    folds: list[Fold] = []
    for i in range(n_folds):
        start = i * window
        end = n if i == n_folds - 1 else start + window
        fold_prices = prices.iloc[start:end]
        result = optimize(
            fold_prices,
            base=base,
            objective_kind=objective_kind,
            space=space,
            split=train_frac,
            n_trials=n_trials,
            seed=seed + i,
            dd_cap=dd_cap,
            on_trial=fold_progress,
        )
        completed += n_trials
        if (
            result.test_metrics is None
            or result.test_index_metrics is None
            or result.test_window is None
        ):
            raise ValueError(
                f"fold {i} produced no test slice; reduce n_folds or train_frac for {n} rows"
            )
        train_excess = result.train_metrics["cagr"] - result.train_index_metrics["cagr"]
        test_excess = result.test_metrics["cagr"] - result.test_index_metrics["cagr"]
        folds.append(
            Fold(
                train_window=result.train_window,
                test_window=result.test_window,
                config=result.best,
                train_excess_cagr=train_excess,
                test_excess_cagr=test_excess,
                result=result,
            )
        )

    final = optimize(
        prices,
        base=base,
        objective_kind=objective_kind,
        space=space,
        split=1.0,
        n_trials=n_trials,
        seed=seed,
        dd_cap=dd_cap,
        on_trial=fold_progress,
    )

    mean_test_excess = sum(f.test_excess_cagr for f in folds) / len(folds)
    return WalkForwardResult(folds=folds, mean_test_excess_cagr=mean_test_excess, final=final)

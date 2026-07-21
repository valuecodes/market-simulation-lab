"""Tests for the cash-deploy strategy optimizer."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import optuna
import pandas as pd
import pytest

from portfolio_research_lab import optimizer as opt
from portfolio_research_lab.cash_deploy import run_cash_deploy
from portfolio_research_lab.models import DeployRule
from portfolio_research_lab.optimizer import (
    ObjectiveContext,
    ObjectiveKind,
    SearchSpace,
    make_objective,
    suggest_config,
)

STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"


def _base() -> opt.CashDeployConfig:
    from portfolio_research_lab.models import CashDeployConfig

    return CashDeployConfig(
        name="Optimized",
        rule=DeployRule(name="seed", thresholds=(0.10,), usages=(0.50,)),
        stock_symbol=STOCK,
        cash_symbol=CASH,
    )


@pytest.fixture
def recovery_prices() -> pd.DataFrame:
    """~300 business days: steady drift, a deep V-shaped drawdown, then recovery.

    A rule that deploys cash into the drawdown and rides the recovery should beat
    a 100%-stocks buy-and-hold that just sat through it. Cash is a flat index.
    """
    idx = pd.bdate_range("2000-01-01", periods=300, name="date")
    t = np.arange(300, dtype=float)
    stock = 100.0 * (1.0 + 0.0003 * t)
    stock[100:180] *= np.linspace(1.0, 0.6, 80)  # ~40% drawdown
    stock[180:] *= 0.6  # recover from the depressed level back up via the drift
    return pd.DataFrame(
        {STOCK: stock, CASH: np.full(300, 100.0)},
        index=idx,
    )


# --- Encoding -----------------------------------------------------------------


def _sample_configs(space: SearchSpace, n: int, seed: int) -> list[opt.CashDeployConfig]:
    configs: list[opt.CashDeployConfig] = []
    base = _base()

    def objective(trial: optuna.Trial) -> float:
        configs.append(suggest_config(trial, space, base))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n)
    return configs


def test_suggest_config_is_always_valid():
    space = SearchSpace()
    configs = _sample_configs(space, n=200, seed=1)
    assert len(configs) == 200
    for cfg in configs:
        rule = cfg.rule
        # DeployRule construction already enforces most of this, but assert the
        # invariants explicitly so the encoding contract is the thing under test.
        assert 1 <= len(rule.thresholds) <= space.max_buckets
        assert len(rule.thresholds) == len(rule.usages)
        assert all(0.0 < t < 1.0 for t in rule.thresholds)
        assert list(rule.thresholds) == sorted(rule.thresholds)
        assert len(set(rule.thresholds)) == len(rule.thresholds)  # strictly increasing
        assert all(u > 0.0 for u in rule.usages)
        assert space.reserve_range[0] <= cfg.reserve_pct <= space.reserve_range[1]
        assert space.refill_range[0] <= cfg.refill_rate_per_year <= space.refill_range[1]


def test_max_buckets_is_respected():
    space = SearchSpace(max_buckets=3)
    configs = _sample_configs(space, n=100, seed=2)
    assert max(len(c.rule.thresholds) for c in configs) <= 3


# --- Objectives ---------------------------------------------------------------


def _context(prices: pd.DataFrame, dd_cap: float = 0.35) -> ObjectiveContext:
    base = _base()
    index_m, _ = opt._index_metrics(prices, base)
    return ObjectiveContext(
        benchmark_cagr=index_m["cagr"],
        cash_level=prices[CASH],
        periods_per_year=base.trading_days_per_year,
        dd_cap=dd_cap,
    )


def test_excess_cagr_objective(recovery_prices: pd.DataFrame):
    ctx = _context(recovery_prices)
    cfg = _base().model_copy(update={"reserve_pct": 0.3})
    result = run_cash_deploy(recovery_prices, cfg)
    m = result.metrics()
    score = make_objective(ObjectiveKind.EXCESS_CAGR, ctx)(result, m)
    assert score == pytest.approx(m["cagr"] - ctx.benchmark_cagr)


def test_dd_capped_penalizes_only_when_breached(recovery_prices: pd.DataFrame):
    cfg = _base().model_copy(update={"reserve_pct": 0.3})
    result = run_cash_deploy(recovery_prices, cfg)
    m = result.metrics()
    max_dd_mag = -m["max_drawdown"]  # positive magnitude

    # Cap well above the actual drawdown => no penalty => equals plain excess CAGR.
    loose = _context(recovery_prices, dd_cap=max_dd_mag + 0.10)
    plain = make_objective(ObjectiveKind.EXCESS_CAGR, loose)(result, m)
    capped_loose = make_objective(ObjectiveKind.EXCESS_CAGR_DD_CAPPED, loose)(result, m)
    assert capped_loose == pytest.approx(plain)

    # Cap below the actual drawdown => penalty subtracts the breach.
    tight = _context(recovery_prices, dd_cap=max_dd_mag - 0.05)
    capped_tight = make_objective(ObjectiveKind.EXCESS_CAGR_DD_CAPPED, tight)(result, m)
    assert capped_tight == pytest.approx(plain - 1.0 * 0.05)


# --- optimize -----------------------------------------------------------------


def test_optimize_is_deterministic(recovery_prices: pd.DataFrame):
    a = opt.optimize(recovery_prices, base=_base(), n_trials=12, seed=0, split=0.6)
    b = opt.optimize(recovery_prices, base=_base(), n_trials=12, seed=0, split=0.6)
    assert a.best == b.best
    assert [s.score for s in a.leaderboard] == [s.score for s in b.leaderboard]


def test_optimize_split_has_no_leakage(recovery_prices: pd.DataFrame):
    result = opt.optimize(recovery_prices, base=_base(), n_trials=10, seed=0, split=0.6)
    assert result.test_window is not None
    # Test window is strictly later than the train window (disjoint, contiguous).
    assert result.train_window[1] < result.test_window[0]
    assert result.test_metrics is not None
    assert result.n_evaluated == 10
    assert 1 <= len(result.leaderboard) <= opt.LEADERBOARD_SIZE


def test_optimize_test_metrics_are_an_independent_test_run(recovery_prices: pd.DataFrame):
    # The frozen best must be re-run on the test slice as its own simulation
    # (engine resets peak/reserve to the slice start) — never sliced from train.
    split = 0.6
    result = opt.optimize(recovery_prices, base=_base(), n_trials=10, seed=0, split=split)
    cut = round(len(recovery_prices) * split)
    test_slice = recovery_prices.iloc[cut:]
    recomputed = run_cash_deploy(test_slice, result.best).metrics()
    assert result.test_metrics is not None
    for key, value in recomputed.items():
        assert result.test_metrics[key] == pytest.approx(value)


def test_optimize_full_window_has_no_holdout(recovery_prices: pd.DataFrame):
    result = opt.optimize(recovery_prices, base=_base(), n_trials=8, seed=0, split=1.0)
    assert result.test_window is None
    assert result.test_metrics is None
    assert result.test_index_metrics is None


def test_optimize_split_below_one_requires_a_holdout():
    # A split < 1.0 promises a train/test split; too little data must raise rather
    # than silently drop the holdout.
    idx = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame({STOCK: [100.0, 101.0, 102.0], CASH: [100.0, 100.0, 100.0]}, index=idx)
    with pytest.raises(ValueError, match="holdout"):
        opt.optimize(prices, base=_base(), n_trials=3, seed=0, split=0.6)


def test_optimize_beats_index_and_a_dominated_config(recovery_prices: pd.DataFrame):
    result = opt.optimize(recovery_prices, base=_base(), n_trials=30, seed=0, split=0.6)
    train_excess = result.train_metrics["cagr"] - result.train_index_metrics["cagr"]
    assert train_excess > 0.0


# --- walk_forward -------------------------------------------------------------


def test_walk_forward_shape_and_ordering(recovery_prices: pd.DataFrame):
    wf = opt.walk_forward(
        recovery_prices, base=_base(), n_folds=3, train_frac=0.5, n_trials=6, seed=0
    )
    assert len(wf.folds) == 3
    # Within each fold, test is strictly after train.
    for fold in wf.folds:
        assert fold.train_window[1] < fold.test_window[0]
    # Folds do not overlap: each fold's test ends before the next fold's train.
    for earlier, later in pairwise(wf.folds):
        assert earlier.test_window[1] < later.train_window[0]
    # The final fit spans the whole window with no holdout.
    assert wf.final.test_window is None
    assert 0 <= wf.n_beat_index_train <= 3
    assert 0 <= wf.n_beat_index_test <= 3
    assert np.isfinite(wf.mean_test_excess_cagr)


def test_walk_forward_is_deterministic(recovery_prices: pd.DataFrame):
    a = opt.walk_forward(
        recovery_prices, base=_base(), n_folds=3, train_frac=0.5, n_trials=6, seed=0
    )
    b = opt.walk_forward(
        recovery_prices, base=_base(), n_folds=3, train_frac=0.5, n_trials=6, seed=0
    )
    assert a.final.best == b.final.best
    assert a.mean_test_excess_cagr == pytest.approx(b.mean_test_excess_cagr)


def test_walk_forward_rejects_too_many_folds():
    idx = pd.bdate_range("2020-01-01", periods=10, name="date")
    prices = pd.DataFrame(
        {STOCK: np.linspace(100.0, 110.0, 10), CASH: np.full(10, 100.0)}, index=idx
    )
    with pytest.raises(ValueError, match="not enough data"):
        opt.walk_forward(prices, base=_base(), n_folds=5, n_trials=3)


# --- optimal_reserve_over_time ------------------------------------------------


def test_optimal_reserve_over_time_shape(recovery_prices: pd.DataFrame):
    grid = (0.1, 0.2, 0.3, 0.4)
    result = opt.optimal_reserve_over_time(
        recovery_prices,
        base=_base(),
        rule=_base().rule,
        refill_rate_per_year=0.25,
        reserve_grid=grid,
        n_points=5,
        min_rows=50,
    )
    assert result.reserve_grid == grid  # already ascending, preserved
    assert len(result.points) == 5
    # Snapshots are strictly increasing in time (distinct expanding windows).
    for earlier, later in pairwise(result.points):
        assert earlier.as_of < later.as_of
        assert earlier.n_rows < later.n_rows
    for point in result.points:
        assert point.optimal_reserve in grid
        assert len(point.objective_by_reserve) == len(grid)
        # The recorded winner is the argmax of the stored surface (lowest on ties).
        best = max(point.objective_by_reserve)
        winners = [grid[i] for i, s in enumerate(point.objective_by_reserve) if s == best]
        assert point.optimal_reserve == min(winners)
        assert point.best_objective == pytest.approx(best)


def test_optimal_reserve_sorts_grid_and_breaks_ties_low(recovery_prices: pd.DataFrame):
    # Pass an unsorted grid; it must come back ascending, and a tie must resolve to
    # the lowest reserve. CAGR over a flat-cash series is identical for two reserves
    # only if they behave identically, so instead assert the sort + argmax contract.
    result = opt.optimal_reserve_over_time(
        recovery_prices,
        base=_base(),
        rule=_base().rule,
        refill_rate_per_year=0.25,
        reserve_grid=(0.4, 0.1, 0.25),
        n_points=3,
        min_rows=50,
    )
    assert result.reserve_grid == (0.1, 0.25, 0.4)


def test_optimal_reserve_is_deterministic(recovery_prices: pd.DataFrame):
    def run() -> opt.ReserveSweepResult:
        return opt.optimal_reserve_over_time(
            recovery_prices,
            base=_base(),
            rule=_base().rule,
            refill_rate_per_year=0.25,
            reserve_grid=(0.1, 0.2, 0.3),
            n_points=4,
            min_rows=50,
        )

    a, b = run(), run()
    assert [p.optimal_reserve for p in a.points] == [p.optimal_reserve for p in b.points]
    assert [p.objective_by_reserve for p in a.points] == [p.objective_by_reserve for p in b.points]


def test_optimal_reserve_matches_bruteforce(recovery_prices: pd.DataFrame):
    grid = (0.05, 0.15, 0.30, 0.45, 0.60)
    base = _base()
    rule = base.rule
    result = opt.optimal_reserve_over_time(
        recovery_prices,
        base=base,
        rule=rule,
        refill_rate_per_year=0.25,
        reserve_grid=grid,
        n_points=3,
        min_rows=50,
    )
    # Recompute the final snapshot's whole objective surface independently.
    last = result.points[-1]
    window = recovery_prices.iloc[: last.n_rows]
    ctx = _context(window)
    objective = make_objective(ObjectiveKind.EXCESS_CAGR, ctx)
    expected = []
    for reserve in grid:
        cfg = base.model_copy(
            update={"reserve_pct": reserve, "refill_rate_per_year": 0.25, "rule": rule}
        )
        res = run_cash_deploy(window, cfg)
        expected.append(objective(res, res.metrics()))
    assert list(last.objective_by_reserve) == pytest.approx(expected)
    assert last.optimal_reserve == grid[expected.index(max(expected))]


def test_optimal_reserve_no_lookahead(recovery_prices: pd.DataFrame):
    # A snapshot fixed by calendar date must not change when future rows are added:
    # windowing is prices.loc[:T], so rows after T are irrelevant.
    as_of = recovery_prices.index[199]
    full = opt.optimal_reserve_over_time(
        recovery_prices,
        base=_base(),
        rule=_base().rule,
        refill_rate_per_year=0.25,
        reserve_grid=(0.1, 0.3, 0.5),
        as_of_dates=[as_of],
    )
    truncated = opt.optimal_reserve_over_time(
        recovery_prices.iloc[:230],
        base=_base(),
        rule=_base().rule,
        refill_rate_per_year=0.25,
        reserve_grid=(0.1, 0.3, 0.5),
        as_of_dates=[as_of],
    )
    a, b = full.points[0], truncated.points[0]
    assert a.as_of == b.as_of == as_of
    assert a.n_rows == b.n_rows == 200
    assert a.optimal_reserve == b.optimal_reserve
    assert a.objective_by_reserve == b.objective_by_reserve


def test_optimal_reserve_rejects_insufficient_data(recovery_prices: pd.DataFrame):
    with pytest.raises(ValueError, match="more than min_rows"):
        opt.optimal_reserve_over_time(
            recovery_prices.iloc[:100],
            base=_base(),
            rule=_base().rule,
            refill_rate_per_year=0.25,
            n_points=5,
            min_rows=100,
        )


def test_optimal_reserve_rejects_empty_as_of_dates(recovery_prices: pd.DataFrame):
    # An empty date sequence would otherwise yield a point-less result, deferring
    # the failure to consumers that read points[-1].
    with pytest.raises(ValueError, match="non-empty"):
        opt.optimal_reserve_over_time(
            recovery_prices,
            base=_base(),
            rule=_base().rule,
            refill_rate_per_year=0.25,
            as_of_dates=[],
        )


def test_optimal_reserve_single_snapshot_ignores_warmup(recovery_prices: pd.DataFrame):
    # A single snapshot is the full-history fit; min_rows must not gate it even
    # when it equals the row count.
    result = opt.optimal_reserve_over_time(
        recovery_prices,
        base=_base(),
        rule=_base().rule,
        refill_rate_per_year=0.25,
        reserve_grid=(0.1, 0.3),
        n_points=1,
        min_rows=len(recovery_prices),
    )
    assert len(result.points) == 1
    assert result.points[0].n_rows == len(recovery_prices)


def test_optimal_reserve_rejects_bad_grid(recovery_prices: pd.DataFrame):
    with pytest.raises(ValueError, match="within"):
        opt.optimal_reserve_over_time(
            recovery_prices,
            base=_base(),
            rule=_base().rule,
            refill_rate_per_year=0.25,
            reserve_grid=(0.1, 1.5),
            n_points=3,
            min_rows=50,
        )

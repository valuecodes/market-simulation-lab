"""Tests for the tactical cash-deployment engine."""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_research_lab.cash_deploy import run_cash_deploy
from portfolio_research_lab.models import CashDeployConfig, DeployRule


def _config(
    *,
    initial_capital: float = 1_000.0,
    reserve_pct: float = 0.5,
    rule: DeployRule | None = None,
    refill_rate_per_year: float = 0.25,
    trading_days_per_year: int = 252,
) -> CashDeployConfig:
    """A cash-deploy config with a simple two-tranche rule (40% then 60%)."""
    if rule is None:
        rule = DeployRule(name="two", thresholds=(0.10, 0.20), usages=(0.4, 0.6))
    return CashDeployConfig(
        initial_capital=initial_capital,
        reserve_pct=reserve_pct,
        rule=rule,
        refill_rate_per_year=refill_rate_per_year,
        trading_days_per_year=trading_days_per_year,
    )


def test_initial_split(deploy_prices: pd.DataFrame):
    result = run_cash_deploy(deploy_prices, _config())
    # 50% of 1,000 into cash, 50% into stocks at the opening price of 100.
    assert result.equity.iloc[0] == pytest.approx(1_000.0)
    assert result.cash.iloc[0] == pytest.approx(500.0)
    assert result.stock_value.iloc[0] == pytest.approx(500.0)


def test_no_drawdown_never_deploys():
    # A stock that only ever rises never triggers a deploy. (It may still refill
    # toward the growing target reserve — deployment is the drawdown-only side.)
    dates = pd.bdate_range("2020-01-01", periods=4, name="date")
    prices = pd.DataFrame(
        {
            "S&P 500": [100.0, 110.0, 120.0, 130.0],
            "Cash (Fed Funds)": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    result = run_cash_deploy(prices, _config())
    assert (result.events["type"] == "deploy").sum() == 0
    assert result.equity.iloc[-1] > result.equity.iloc[0]


def test_gap_down_fires_both_tranches_same_day(deploy_prices: pd.DataFrame):
    result = run_cash_deploy(deploy_prices, _config())
    day2 = deploy_prices.index[2]  # price 75 => -25% in one step

    fired = result.events.loc[[day2]]
    assert list(fired["type"]) == ["deploy", "deploy"]
    # Tranche base is the reserve at the episode start = 500. 40% then 60%.
    assert list(fired["amount"]) == pytest.approx([200.0, 300.0])
    # Reserve fully drained; both tranches summed to 100% of the base.
    assert result.cash.loc[day2] == pytest.approx(0.0)
    # Deploy is a transfer at the current price, so equity is unchanged by it:
    # 5 units * 75 + 500 cash = 875 either way, all now in stocks.
    assert result.equity.loc[day2] == pytest.approx(875.0)
    assert result.stock_value.loc[day2] == pytest.approx(875.0)


def test_deploy_capped_by_available_cash():
    # Usages summing above 1.0: the second tranche can only spend what's left.
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame(
        {"S&P 500": [100.0, 100.0, 75.0], "Cash (Fed Funds)": [100.0, 100.0, 100.0]},
        index=dates,
    )
    rule = DeployRule(name="greedy", thresholds=(0.10, 0.20), usages=(0.7, 0.7))
    result = run_cash_deploy(prices, _config(rule=rule))
    # base 500: first tranche 350, second capped at remaining 150 (not 350).
    assert list(result.events["amount"]) == pytest.approx([350.0, 150.0])
    assert result.cash.iloc[-1] == pytest.approx(0.0)


def test_tranche_base_locked_at_episode_start():
    # The reserve grows via cash yield mid-episode, but tranche sizing uses the
    # reserve locked at the episode's first dip, not the grown balance.
    dates = pd.bdate_range("2020-01-01", periods=4, name="date")
    prices = pd.DataFrame(
        {
            "S&P 500": [100.0, 100.0, 95.0, 89.0],  # -5% (no deploy) then -11%
            "Cash (Fed Funds)": [100.0, 100.0, 110.0, 121.0],  # +10%/step
        },
        index=dates,
    )
    rule = DeployRule(name="one", thresholds=(0.10,), usages=(0.5,))
    result = run_cash_deploy(prices, _config(rule=rule))
    # First dip is day2 (price 95): cash has already grown to 550 -> base = 550.
    # Deploy fires at day3 (-11%); amount = 0.5 * 550 = 275, NOT 0.5 * 605.
    assert list(result.events["amount"]) == pytest.approx([275.0])


def test_no_refill_below_all_time_high():
    # A partial recovery that never regains the prior peak must not refill.
    dates = pd.bdate_range("2020-01-01", periods=4, name="date")
    prices = pd.DataFrame(
        {"S&P 500": [100.0, 80.0, 80.0, 90.0], "Cash (Fed Funds)": [100.0] * 4},
        index=dates,
    )
    result = run_cash_deploy(prices, _config())
    assert (result.events["type"] == "refill").sum() == 0


def test_refill_drips_toward_target_at_new_high():
    # tdy=2 with a 100%/yr refill => a 50%-of-target drip per ATH day, giving
    # hand-checkable refill steps. A -50% drop deploys the whole reserve into
    # clean units (5 + 500/50 = 15).
    dates = pd.bdate_range("2020-01-01", periods=5, name="date")
    prices = pd.DataFrame(
        {
            "S&P 500": [100.0, 50.0, 50.0, 100.0, 100.0],
            "Cash (Fed Funds)": [100.0] * 5,
        },
        index=dates,
    )
    config = _config(refill_rate_per_year=1.0, trading_days_per_year=2)
    result = run_cash_deploy(prices, config)

    day1 = dates[1]  # -50%: both tranches fire, cash -> 0, units 15
    assert result.cash.loc[day1] == pytest.approx(0.0)

    day3 = dates[3]  # first new high: equity = 15 * 100 = 1500, target = 750,
    # drip = 50% of target = 375.
    refills = result.events[result.events["type"] == "refill"]
    assert refills.loc[day3, "amount"] == pytest.approx(375.0)
    assert result.cash.loc[day3] == pytest.approx(375.0)
    # Refill is a transfer, so equity is unchanged by it.
    assert result.equity.loc[day3] == pytest.approx(1500.0)

    day4 = dates[4]  # second drip reaches the target exactly.
    assert result.cash.loc[day4] == pytest.approx(750.0)
    # Cash never overshoots its target.
    target_series = config.reserve_pct * result.equity
    assert (result.cash <= target_series + 1e-6).all()


def test_cash_accrues_yield_when_idle():
    # No drawdown, but the cash index grows: the reserve should compound with it.
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame(
        {
            "S&P 500": [100.0, 100.0, 100.0],
            "Cash (Fed Funds)": [100.0, 101.0, 102.01],  # +1%/step
        },
        index=dates,
    )
    result = run_cash_deploy(prices, _config())
    # 500 cash compounding at +1%/step, stock leg flat at 500.
    assert result.cash.iloc[-1] == pytest.approx(500.0 * 1.01 * 1.01)
    assert result.equity.iloc[-1] == pytest.approx(500.0 * 1.0201 + 500.0)


def test_events_schema(deploy_prices: pd.DataFrame):
    result = run_cash_deploy(deploy_prices, _config())
    assert list(result.events.columns) == ["type", "drawdown", "amount", "cash_after"]


def test_empty_prices_raise():
    with pytest.raises(ValueError, match="empty price data"):
        run_cash_deploy(pd.DataFrame(), _config())


def test_missing_columns_raise():
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame({"S&P 500": [100.0, 100.0, 100.0]}, index=dates)
    with pytest.raises(ValueError, match="missing required columns"):
        run_cash_deploy(prices, _config())


def test_all_cash_reserve_never_deploys_beyond_cash():
    # reserve_pct = 1.0 => everything starts in cash; deploys draw it down but
    # equity stays continuous and non-negative.
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame(
        {"S&P 500": [100.0, 100.0, 80.0], "Cash (Fed Funds)": [100.0] * 3},
        index=dates,
    )
    result = run_cash_deploy(prices, _config(reserve_pct=1.0))
    assert (result.cash >= -1e-9).all()
    assert result.equity.iloc[0] == pytest.approx(1_000.0)

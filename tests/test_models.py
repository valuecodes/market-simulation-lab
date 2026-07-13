"""Tests for the configuration models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from portfolio_research_lab.models import (
    PRESET_RULES,
    CashDeployConfig,
    DeployRule,
    StrategyConfig,
)


def test_valid_config():
    config = StrategyConfig(allocations={"A": 0.6, "B": 0.4})
    assert config.symbols == ["A", "B"]
    assert config.initial_capital == 10_000.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        StrategyConfig(allocations={"A": 0.6, "B": 0.6})


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_weight_rejected(bad):
    # NaN/inf must not slip past the positive-weight and sum checks.
    with pytest.raises(ValidationError, match="positive number"):
        StrategyConfig(allocations={"A": bad})


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_initial_capital_rejected(bad):
    with pytest.raises(ValidationError):
        StrategyConfig(allocations={"A": 1.0}, initial_capital=bad)


def test_negative_weight_rejected():
    with pytest.raises(ValidationError, match="must be a positive number"):
        StrategyConfig(allocations={"A": 1.2, "B": -0.2})


def test_non_positive_capital_rejected():
    with pytest.raises(ValidationError):
        StrategyConfig(allocations={"A": 1.0}, initial_capital=0)


def test_from_weights_normalizes():
    config = StrategyConfig.from_weights({"A": 30.0, "B": 10.0}, normalize=True)
    assert config.allocations["A"] == pytest.approx(0.75)
    assert config.allocations["B"] == pytest.approx(0.25)


def test_from_weights_rejects_zero_total():
    with pytest.raises(ValueError, match="sum to <= 0"):
        StrategyConfig.from_weights({"A": 0.0}, normalize=True)


def test_rebalance_frequency_defaults_to_none():
    config = StrategyConfig(allocations={"A": 1.0})
    assert config.rebalance_frequency is None


@pytest.mark.parametrize("freq", ["monthly", "quarterly", "annually"])
def test_valid_rebalance_frequencies(freq: str):
    config = StrategyConfig(allocations={"A": 1.0}, rebalance_frequency=freq)
    assert config.rebalance_frequency == freq


def test_unknown_rebalance_frequency_rejected():
    with pytest.raises(ValidationError, match="rebalance_frequency must be one of"):
        StrategyConfig(allocations={"A": 1.0}, rebalance_frequency="daily")


# --- DeployRule ----------------------------------------------------------


def test_deploy_rule_valid():
    rule = DeployRule(name="r", thresholds=(0.1, 0.2, 0.3), usages=(0.5, 0.3, 0.2))
    assert rule.usage_sum == pytest.approx(1.0)


def test_deploy_rule_thresholds_must_increase():
    with pytest.raises(ValidationError, match="strictly increasing"):
        DeployRule(name="r", thresholds=(0.2, 0.1), usages=(0.5, 0.5))


def test_deploy_rule_thresholds_must_be_in_unit_interval():
    with pytest.raises(ValidationError, match=r"in \(0, 1\)"):
        DeployRule(name="r", thresholds=(0.1, 1.5), usages=(0.5, 0.5))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_deploy_rule_rejects_non_finite_threshold(bad: float):
    with pytest.raises(ValidationError):
        DeployRule(name="r", thresholds=(0.1, bad), usages=(0.5, 0.5))


def test_deploy_rule_usages_must_be_positive():
    with pytest.raises(ValidationError, match="must be a positive number"):
        DeployRule(name="r", thresholds=(0.1, 0.2), usages=(0.5, -0.1))


def test_deploy_rule_lengths_must_match():
    with pytest.raises(ValidationError, match="equal length"):
        DeployRule(name="r", thresholds=(0.1, 0.2), usages=(1.0,))


# --- CashDeployConfig ----------------------------------------------------

_RULE = DeployRule(name="r", thresholds=(0.1, 0.2), usages=(0.5, 0.5))


def test_cash_deploy_config_defaults():
    config = CashDeployConfig(rule=_RULE)
    assert config.reserve_pct == pytest.approx(0.30)
    assert config.refill_rate_per_year == pytest.approx(0.25)
    assert config.trading_days_per_year == 252


@pytest.mark.parametrize("reserve", [-0.1, 1.1])
def test_cash_deploy_config_reserve_pct_bounds(reserve: float):
    with pytest.raises(ValidationError):
        CashDeployConfig(rule=_RULE, reserve_pct=reserve)


def test_cash_deploy_config_rejects_equal_symbols():
    with pytest.raises(ValidationError, match="must be different"):
        CashDeployConfig(rule=_RULE, stock_symbol="X", cash_symbol="X")


def test_cash_deploy_config_rejects_non_positive_capital():
    with pytest.raises(ValidationError):
        CashDeployConfig(rule=_RULE, initial_capital=0)


# --- Presets -------------------------------------------------------------


def test_preset_rules_present_and_valid():
    assert set(PRESET_RULES) == {"User rule", "Growth optimum", "Risk optimum", "Recommended"}
    for rule in PRESET_RULES.values():
        assert len(rule.thresholds) == len(rule.usages) == 5
        assert rule.usage_sum == pytest.approx(1.0)

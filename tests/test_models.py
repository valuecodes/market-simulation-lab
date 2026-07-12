"""Tests for the configuration models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from portfolio_research_lab.models import StrategyConfig


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

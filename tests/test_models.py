"""Tests for the configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from portfolio_research_lab.models import StrategyConfig


def test_valid_config():
    config = StrategyConfig(allocations={"A": 0.6, "B": 0.4})
    assert config.symbols == ["A", "B"]
    assert config.initial_capital == 10_000.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        StrategyConfig(allocations={"A": 0.6, "B": 0.6})


def test_negative_weight_rejected():
    with pytest.raises(ValidationError, match="must be positive"):
        StrategyConfig(allocations={"A": 1.2, "B": -0.2})


def test_non_positive_capital_rejected():
    with pytest.raises(ValidationError):
        StrategyConfig(allocations={"A": 1.0}, initial_capital=0)


def test_from_weights_normalizes():
    config = StrategyConfig.from_weights({"A": 30, "B": 10}, normalize=True)
    assert config.allocations["A"] == pytest.approx(0.75)
    assert config.allocations["B"] == pytest.approx(0.25)


def test_from_weights_rejects_zero_total():
    with pytest.raises(ValueError, match="sum to <= 0"):
        StrategyConfig.from_weights({"A": 0.0}, normalize=True)

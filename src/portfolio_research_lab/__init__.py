"""Portfolio Research Lab.

A local-first toolkit for backtesting and simulating investment strategies.

The package is split into a reusable simulation engine (this package) and a
Streamlit interface (see the ``app/`` directory). Nothing in this package
imports Streamlit, so the engine can be used from notebooks, scripts or tests.
"""

from __future__ import annotations

from portfolio_research_lab.cash_deploy import CashDeployResult, run_cash_deploy
from portfolio_research_lab.data import (
    infer_periods_per_year,
    load_price_data,
    load_rate_series,
    parse_price_csv,
    rate_to_index,
)
from portfolio_research_lab.models import CashDeployConfig, DeployRule, StrategyConfig
from portfolio_research_lab.optimizer import (
    ObjectiveKind,
    OptimizationResult,
    ReserveSweepPoint,
    ReserveSweepResult,
    SearchSpace,
    WalkForwardResult,
    optimal_reserve_over_time,
    optimize,
    walk_forward,
)
from portfolio_research_lab.simulator import SimulationResult, run_simulation
from portfolio_research_lab.strategies import BuyAndHold, Strategy

__all__ = [
    "BuyAndHold",
    "CashDeployConfig",
    "CashDeployResult",
    "DeployRule",
    "ObjectiveKind",
    "OptimizationResult",
    "ReserveSweepPoint",
    "ReserveSweepResult",
    "SearchSpace",
    "SimulationResult",
    "Strategy",
    "StrategyConfig",
    "WalkForwardResult",
    "infer_periods_per_year",
    "load_price_data",
    "load_rate_series",
    "optimal_reserve_over_time",
    "optimize",
    "parse_price_csv",
    "rate_to_index",
    "run_cash_deploy",
    "run_simulation",
    "walk_forward",
]

__version__ = "0.1.0"

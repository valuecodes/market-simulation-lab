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
from portfolio_research_lab.models import (
    CashDeployConfig,
    DeployRule,
    StrategyConfig,
    TimingConfig,
)
from portfolio_research_lab.optimizer import (
    ObjectiveKind,
    OptimizationResult,
    SearchSpace,
    WalkForwardResult,
    optimize,
    walk_forward,
)
from portfolio_research_lab.simulator import SimulationResult, run_simulation
from portfolio_research_lab.strategies import BuyAndHold, Strategy
from portfolio_research_lab.timing import TimingResult, run_ma_timing

__all__ = [
    "BuyAndHold",
    "CashDeployConfig",
    "CashDeployResult",
    "DeployRule",
    "ObjectiveKind",
    "OptimizationResult",
    "SearchSpace",
    "SimulationResult",
    "Strategy",
    "StrategyConfig",
    "TimingConfig",
    "TimingResult",
    "WalkForwardResult",
    "infer_periods_per_year",
    "load_price_data",
    "load_rate_series",
    "optimize",
    "parse_price_csv",
    "rate_to_index",
    "run_cash_deploy",
    "run_ma_timing",
    "run_simulation",
    "walk_forward",
]

__version__ = "0.1.0"

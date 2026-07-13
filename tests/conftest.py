"""Shared test fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def flat_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=6, name="date")


@pytest.fixture
def two_asset_prices(flat_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A tiny, hand-checkable price panel.

    ``UP`` doubles over the window; ``FLAT`` never moves; ``BENCH`` grows 50%.
    """
    return pd.DataFrame(
        {
            "UP": [100.0, 110.0, 120.0, 130.0, 140.0, 200.0],
            "FLAT": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            "BENCH": [100.0, 100.0, 100.0, 100.0, 100.0, 150.0],
        },
        index=flat_dates,
    )


@pytest.fixture
def deploy_prices(flat_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A hand-checkable stocks + cash panel for the cash-deploy engine.

    Six business days. ``S&P 500`` sits at its opening high, gaps down 25% (in
    one step, so a two-tranche rule with thresholds at 10% and 20% fires both
    tranches at once), holds, then recovers to the old high and makes a new one.
    ``Cash (Fed Funds)`` is a flat index (per-step growth factor 1.0), so the
    reserve earns no interest and the arithmetic stays exact.
    """
    return pd.DataFrame(
        {
            "S&P 500": [100.0, 100.0, 75.0, 75.0, 100.0, 125.0],
            "Cash (Fed Funds)": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=flat_dates,
    )

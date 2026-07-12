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

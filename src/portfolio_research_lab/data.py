"""Loading and generating price data.

Price data is represented as a :class:`pandas.DataFrame` indexed by date, with
one column of closing prices per asset. This "wide" shape is what the simulator
and metrics consume throughout the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "date"


def load_price_data(path: str | Path, *, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Load a wide-format price CSV.

    The CSV must contain a date column plus one column of prices per asset::

        date,STOCKS,BONDS,GOLD
        2019-01-02,100.0,100.0,100.0
        2019-01-03,100.8,100.1,99.4
        ...

    Returns a DataFrame indexed by a sorted :class:`~pandas.DatetimeIndex` with
    float price columns. Missing values are forward-filled (holidays / gaps).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"price data file not found: {path}")

    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"expected a {date_column!r} column, found {list(frame.columns)}")

    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.set_index(date_column).sort_index()

    price_columns = frame.columns.tolist()
    if not price_columns:
        raise ValueError("price data has no asset columns")

    frame = frame.astype(float)
    frame = frame.ffill().dropna(how="any")
    if frame.empty:
        raise ValueError("price data is empty after cleaning")
    return frame


def generate_synthetic_prices(
    *,
    assets: dict[str, tuple[float, float]] | None = None,
    start: str = "2019-01-01",
    periods: int = 252 * 5,
    initial_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic daily prices via geometric Brownian motion.

    Parameters
    ----------
    assets:
        Mapping of symbol to ``(annual_drift, annual_volatility)``. Defaults to a
        small multi-asset universe plus a blended benchmark.
    start:
        First business day of the series.
    periods:
        Number of business days to generate (default ~5 years).
    initial_price:
        Starting price for every asset.
    seed:
        Seed for the random generator so output is reproducible.
    """
    if assets is None:
        assets = {
            "STOCKS": (0.09, 0.18),
            "BONDS": (0.03, 0.05),
            "GOLD": (0.04, 0.15),
            "BENCHMARK": (0.06, 0.11),
        }

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods, name=DATE_COLUMN)
    trading_days = 252
    dt = 1.0 / trading_days

    columns: dict[str, np.ndarray] = {}
    for symbol, (drift, vol) in assets.items():
        shocks = rng.standard_normal(periods)
        # Log returns of a GBM: (mu - 0.5*sigma^2) dt + sigma sqrt(dt) Z
        log_returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
        log_returns[0] = 0.0  # anchor the first observation at the initial price
        prices = initial_price * np.exp(np.cumsum(log_returns))
        columns[symbol] = np.round(prices, 4)

    return pd.DataFrame(columns, index=dates)

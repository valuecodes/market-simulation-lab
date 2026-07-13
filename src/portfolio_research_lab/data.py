"""Loading price and rate data.

Price data is represented as a :class:`pandas.DataFrame` indexed by date, with
one column of closing prices per asset. This "wide" shape is what the simulator
and metrics consume throughout the package. This module also loads interest-rate
series and converts them into cash-account growth indices (see
:func:`rate_to_index`) so a rate can be simulated as a holdable asset.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "date"

# Guardrails for *untrusted* CSV input (uploads). The local-file loaders are
# considered trusted and are not subject to these size bounds; the data-integrity
# checks in ``_prepare_price_frame`` apply to all sources.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_UPLOAD_ROWS = 200_000
MAX_UPLOAD_COLUMNS = 100
MAX_SYMBOL_LENGTH = 64

# Median spacing (in days) between observations -> periods per year, used to
# annualise metrics. Anything sparser than quarterly is treated as annual.
_FREQUENCY_TABLE: tuple[tuple[float, int], ...] = (
    (4, 252),  # daily (business days: 1-3 day gaps across weekends)
    (10, 52),  # weekly
    (45, 12),  # monthly
    (100, 4),  # quarterly
)
_DEFAULT_PERIODS_PER_YEAR = 1  # annual or sparser


def load_price_data(path: str | Path, *, date_column: str | None = None) -> pd.DataFrame:
    """Load a wide-format price CSV.

    The CSV must contain a date column plus one column of prices per asset::

        date,STOCKS,BONDS,GOLD
        2019-01-02,100.0,100.0,100.0
        2019-01-03,100.8,100.1,99.4
        ...

    The loader is tolerant of common real-world quirks: a UTF-8 byte-order mark,
    a differently-cased date column (``Date``/``DATE``), and US ``MM/DD/YYYY``
    dates. If ``date_column`` is ``None`` it is auto-detected.

    Returns a DataFrame indexed by a sorted :class:`~pandas.DatetimeIndex` with
    float price columns. Missing values are forward-filled (holidays / gaps).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"price data file not found: {path}")

    # utf-8-sig transparently strips a byte-order mark if present.
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return _prepare_price_frame(frame, date_column)


def parse_price_csv(data: str | bytes, *, date_column: str | None = None) -> pd.DataFrame:
    """Parse wide-format price CSV text or bytes (e.g. from a file upload).

    Applies the same cleaning rules as :func:`load_price_data`, plus size
    guardrails suitable for *untrusted* input: the payload, row count, column
    count and asset-name length are all bounded to prevent a huge or malicious
    upload from exhausting memory. Raises :class:`ValueError` if a limit is hit.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB size limit")

    # nrows caps parsing work even before the row-count check below.
    frame = pd.read_csv(BytesIO(raw), encoding="utf-8-sig", nrows=MAX_UPLOAD_ROWS + 1)
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"upload exceeds the {MAX_UPLOAD_ROWS} row limit")
    if frame.shape[1] > MAX_UPLOAD_COLUMNS:
        raise ValueError(f"upload exceeds the {MAX_UPLOAD_COLUMNS} column limit")
    return _prepare_price_frame(frame, date_column, max_symbol_length=MAX_SYMBOL_LENGTH)


def load_rate_series(path: str | Path, *, rate_column: str | None = None) -> pd.Series:
    """Load a single interest-rate series from a CSV.

    The CSV must contain a date column plus exactly one rate column, e.g. the
    federal funds rate::

        Date,Value
        07/01/1954,1.13
        07/02/1954,1.25
        ...

    The date column is auto-detected (or named via ``rate_column``'s counterpart
    logic) with the same tolerance as :func:`load_price_data`. Values are
    interpreted as *annualised percentages* (``1.13`` means 1.13%). Returns a
    :class:`pandas.Series` indexed by a sorted :class:`~pandas.DatetimeIndex`,
    forward-filled over gaps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rate data file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    date_col = _resolve_date_column(frame, None)

    value_cols = [c for c in frame.columns if c != date_col]
    if rate_column is not None:
        if rate_column not in value_cols:
            raise ValueError(f"expected a {rate_column!r} column, found {value_cols}")
        value_col = rate_column
    elif len(value_cols) == 1:
        value_col = value_cols[0]
    else:
        raise ValueError(f"expected exactly one rate column, found {value_cols}")

    frame[date_col] = pd.to_datetime(frame[date_col])
    series = frame.set_index(date_col)[value_col].sort_index().astype(float).ffill().dropna()
    series.index.name = DATE_COLUMN
    series.name = value_col
    if series.empty:
        raise ValueError("rate data is empty after cleaning")
    return series


def rate_to_index(
    rate: pd.Series,
    *,
    base: float = 100.0,
    days_per_year: int = 360,
) -> pd.Series:
    """Convert an annualised %-rate series into a money-market growth index.

    Models a cash account that accrues interest at ``rate`` (an annualised
    percentage, so ``1.13`` means 1.13%). Interest is accrued over the *actual
    number of calendar days* between observations, which keeps the index correct
    even when the source series has gaps.

    For each date ``t`` after the first, the growth factor applies the previous
    period's rate over the elapsed days::

        factor[t] = 1 + rate[t-1] / 100 * days_between / days_per_year

    and the index is ``base * factor.cumprod()`` with the first point pinned to
    ``base``. ``days_per_year`` defaults to 360 to match the actual/360
    money-market convention used to quote the federal funds rate.
    """
    if rate.empty:
        raise ValueError("cannot build an index from an empty rate series")

    days = rate.index.to_series().diff().dt.days
    factor = 1.0 + rate.shift(1) / 100.0 * days / days_per_year
    factor.iloc[0] = 1.0  # first point: no prior period to accrue over
    return base * factor.cumprod()


def load_stocks_cash(
    sp500_path: str | Path,
    fed_funds_path: str | Path,
    *,
    stock_name: str = "S&P 500",
    cash_name: str = "Cash (Fed Funds)",
) -> pd.DataFrame:
    """Build the two-asset stocks + cash price frame used across the app.

    Loads daily S&P 500 closes and turns the federal funds rate into a
    compounding money-market index via :func:`rate_to_index`, then joins them on
    the S&P trading days. The cash index is built on the fed funds' native
    (calendar) dates first, so weekend interest accrual is baked into each level
    before the join samples it onto trading days. The final ``dropna`` trims the
    S&P-only years before the fed funds series begins (1954).

    Returns a DataFrame with exactly two float columns, ``stock_name`` and
    ``cash_name``, indexed by a sorted :class:`~pandas.DatetimeIndex`.
    """
    if stock_name == cash_name:
        raise ValueError("stock_name and cash_name must be different")
    stocks = load_price_data(sp500_path).rename(columns={"close": stock_name})
    cash = rate_to_index(load_rate_series(fed_funds_path)).rename(cash_name)
    frame = stocks.join(cash, how="left").ffill().dropna(how="any")
    if frame.empty:
        raise ValueError("stocks + cash frame is empty after aligning the two series")
    return frame


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Guess how many observations occur per year from the index spacing.

    Returns 252 (daily), 52 (weekly), 12 (monthly), 4 (quarterly) or 1 (annual).
    Used to annualise returns and volatility for arbitrary data cadences.
    """
    if len(index) < 3:
        return _FREQUENCY_TABLE[0][1]
    median_gap = index.to_series().diff().dropna().dt.days.median()
    for max_gap, periods in _FREQUENCY_TABLE:
        if median_gap <= max_gap:
            return periods
    return _DEFAULT_PERIODS_PER_YEAR


def _prepare_price_frame(
    frame: pd.DataFrame,
    date_column: str | None,
    *,
    max_symbol_length: int | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    resolved = _resolve_date_column(frame, date_column)

    asset_columns = [c for c in frame.columns if c != resolved]
    if not asset_columns:
        raise ValueError("price data has no asset columns")
    duplicates = sorted({c for c in asset_columns if asset_columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"price data has duplicate asset columns: {duplicates}")
    if max_symbol_length is not None:
        too_long = [c for c in asset_columns if len(c) > max_symbol_length]
        if too_long:
            raise ValueError(f"asset name exceeds {max_symbol_length} characters: {too_long[:3]}")

    frame[resolved] = pd.to_datetime(frame[resolved])
    frame = frame.set_index(resolved).sort_index()
    frame.index.name = DATE_COLUMN
    if frame.index.has_duplicates:
        raise ValueError("price data has duplicate dates")

    frame = frame.astype(float)
    frame = frame.ffill().dropna(how="any")
    if len(frame) < 2:
        raise ValueError("price data needs at least two dated observations after cleaning")

    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("price data contains non-finite values (NaN or infinity)")
    if (values <= 0).any():
        raise ValueError("price data must be strictly positive (no zero or negative prices)")
    return frame


def _resolve_date_column(frame: pd.DataFrame, date_column: str | None) -> str:
    columns = list(frame.columns)
    if date_column is not None:
        if date_column in columns:
            return date_column
        raise ValueError(f"expected a {date_column!r} column, found {columns}")
    for column in columns:
        if column.strip().lower() == DATE_COLUMN:
            return column
    raise ValueError(f"could not find a date column (looked for 'date') in {columns}")

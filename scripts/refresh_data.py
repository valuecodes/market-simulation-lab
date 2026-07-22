#!/usr/bin/env python
"""Fetch the market data the app runs on into ``data/``.

This is the repo's data setup/refresh step. A fresh clone has no ``data/``
directory (it is gitignored), so run this once to populate it, and re-run it
whenever you want the latest observations. It writes the two CSVs the app reads:

- ``data/sp500daily.csv`` — daily S&P 500 (``^GSPC``) closes, ``date,close``.
- ``data/fed-funds-rate.csv`` — daily effective federal funds rate,
  ``"Date","Value"``.

Both come from public, unauthenticated endpoints (no API key needed):

- S&P 500: Yahoo Finance's chart API (``query1.finance.yahoo.com``). This is an
  undocumented/unofficial endpoint, but it is the practical way to get the full
  history back to 1927.
- Fed funds: FRED (Federal Reserve Bank of St. Louis), series ``DFF``. Official
  and stable; identical to the macrotrends fed-funds series.

Each freshly fetched file is validated by loading it back through the app's own
loaders (:mod:`portfolio_research_lab.data`) *before* it replaces the existing
file, so a failed or malformed fetch never clobbers good data.

Usage::

    uv run poe refresh-data        # both series
    python scripts/refresh_data.py --sp500-only
    python scripts/refresh_data.py --fed-funds-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC = PROJECT_ROOT / "src"

# Allow running directly (``python scripts/refresh_data.py``) without installing
# the package, by adding the src/ layout to the import path — mirrors app/Home.py.
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_research_lab.data import load_price_data, load_rate_series  # noqa: E402

SP500_PATH = DATA_DIR / "sp500daily.csv"
FED_FUNDS_PATH = DATA_DIR / "fed-funds-rate.csv"

# --- Sources (public endpoints, no credentials) ---------------------------

# S&P 500 (^GSPC) daily closes — Yahoo Finance public chart API. period1 is the
# instrument's firstTradeDate (1927); period2 is filled in at fetch time.
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
SP500_PERIOD1 = -1325583000  # ^GSPC firstTradeDate (1927-12-30, UTC seconds)

# Daily effective federal funds rate (DFF) — FRED, St. Louis Fed.
FRED_DFF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"

# The two hosts disagree on User-Agents, so each fetch sends its own:
#  - Yahoo returns 401 for the default urllib UA; it wants a browser-like one.
#  - FRED's edge (Akamai) resets the connection for browser-spoofing UAs but is
#    happy with urllib's default, so we deliberately send no UA override there.
_YAHOO_USER_AGENT = "Mozilla/5.0"
_HTTP_TIMEOUT = 60  # seconds


def _http_get(url: str, *, user_agent: str | None = None) -> bytes:
    """GET ``url`` and return the response body, raising on any HTTP error.

    When ``user_agent`` is ``None`` the request keeps urllib's default UA.
    """
    headers = {"User-Agent": user_agent} if user_agent else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc


def _commit(
    rows: list[tuple[str, ...]], header: tuple[str, ...], path: Path, *, quote_all: bool
) -> None:
    """Write ``header`` + ``rows`` to a temp CSV, validate it, then replace ``path``.

    Validation loads the freshly written file through the app's own loader so a
    malformed fetch fails loudly and leaves any existing ``path`` untouched.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    fd, tmp_name = tempfile.mkstemp(dir=DATA_DIR, prefix=path.stem + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, quoting=quoting)
            writer.writerow(header)
            writer.writerows(rows)
        _validate(tmp, path)
        tmp.chmod(0o644)  # mkstemp creates 0600; use normal file perms
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _validate(tmp: Path, target: Path) -> None:
    if target == SP500_PATH:
        load_price_data(tmp)
    else:
        load_rate_series(tmp)


def fetch_sp500() -> None:
    """Fetch full ^GSPC daily-close history from Yahoo and write sp500daily.csv."""
    period2 = int(datetime.now(tz=UTC).timestamp())
    url = f"{YAHOO_CHART_URL}?period1={SP500_PERIOD1}&period2={period2}&interval=1d"
    payload = json.loads(_http_get(url, user_agent=_YAHOO_USER_AGENT))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart API error: {chart['error']}")
    result = chart.get("result") or []
    if not result:
        raise RuntimeError("Yahoo chart API returned no result")
    series = result[0]
    timestamps = series.get("timestamp") or []
    closes = series["indicators"]["quote"][0].get("close") or []
    if not timestamps or len(timestamps) != len(closes):
        raise RuntimeError("Yahoo chart API returned malformed timestamp/close arrays")

    rows: list[tuple[str, ...]] = []
    for ts, close in zip(timestamps, closes, strict=True):
        if close is None:  # holidays / missing prints
            continue
        day = datetime.fromtimestamp(ts, tz=UTC).date()
        # Yahoo returns float32-precision closes (e.g. 17.65999984741211); the
        # index level only warrants 2 dp, matching the original data's format.
        rows.append((day.isoformat(), f"{float(close):.2f}"))
    if not rows:
        raise RuntimeError("Yahoo chart API returned no usable close prices")

    _commit(rows, ("date", "close"), SP500_PATH, quote_all=False)
    _summarize("S&P 500", SP500_PATH, rows[0][0], rows[-1][0], len(rows))


def fetch_fed_funds() -> None:
    """Fetch the daily fed funds rate (DFF) from FRED and write fed-funds-rate.csv."""
    text = _http_get(FRED_DFF_URL).decode("utf-8-sig")
    reader = csv.reader(text.splitlines())
    next(reader, None)  # drop FRED's "observation_date,DFF" header

    rows: list[tuple[str, ...]] = []
    for record in reader:
        if len(record) < 2:
            continue
        raw_date, raw_value = record[0].strip(), record[1].strip()
        if not raw_date or raw_value in {"", "."}:  # FRED marks gaps with "."
            continue
        day = datetime.strptime(raw_date, "%Y-%m-%d").date()
        rows.append((day.strftime("%m/%d/%Y"), raw_value))
    if not rows:
        raise RuntimeError("FRED returned no usable fed funds observations")

    _commit(rows, ("Date", "Value"), FED_FUNDS_PATH, quote_all=True)
    _summarize("Fed funds", FED_FUNDS_PATH, rows[0][0], rows[-1][0], len(rows))


def _summarize(label: str, path: Path, first: str, last: str, count: int) -> None:
    rel = path.relative_to(PROJECT_ROOT)
    print(f"{label}: {count:,} rows ({first} → {last}) → {rel}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch S&P 500 + fed funds data into data/.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sp500-only", action="store_true", help="Only refresh the S&P 500 series")
    group.add_argument(
        "--fed-funds-only", action="store_true", help="Only refresh the fed funds series"
    )
    args = parser.parse_args(argv)

    if not args.fed_funds_only:
        fetch_sp500()
    if not args.sp500_only:
        fetch_fed_funds()


if __name__ == "__main__":
    main()

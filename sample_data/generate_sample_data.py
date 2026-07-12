"""Regenerate the bundled synthetic sample price data.

Run from the project root::

    uv run python sample_data/generate_sample_data.py

The output (``sample_data/sample_prices.csv``) is committed so the Streamlit app
works immediately without running this script. Re-run it only if you want to
change the synthetic universe or date range.
"""

from __future__ import annotations

from pathlib import Path

from portfolio_research_lab.data import generate_synthetic_prices

OUTPUT = Path(__file__).parent / "sample_prices.csv"


def main() -> None:
    prices = generate_synthetic_prices()
    prices.to_csv(OUTPUT)
    print(f"Wrote {len(prices)} rows x {prices.shape[1]} assets to {OUTPUT}")


if __name__ == "__main__":
    main()

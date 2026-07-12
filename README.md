# Portfolio Research Lab

A **local-first** Python application for testing investment and portfolio
strategies through historical backtesting and simulation. It runs entirely on
your machine — no database, no accounts, no cloud services.

The project separates a reusable **simulation engine**
(`src/portfolio_research_lab/`) from the **Streamlit interface** (`app/`). The
engine never imports Streamlit, so you can drive it from notebooks, scripts or
tests just as easily as from the UI.

> ⚠️ **Disclaimer**
> This application is provided for **research and educational purposes only**.
> It uses simplified models, and it is **not financial advice**. Nothing here is
> a recommendation to buy or sell any security. Do your own research and consult
> a qualified professional before investing.

## Features

The initial release is a small but working foundation:

- Load historical asset prices from wide-format **CSV files**.
- Test **multi-asset allocations** — e.g. stocks vs. a cash account built from
  the **federal funds rate** (bundled `Stocks + Cash (1954+)` dataset).
- Configure an **initial portfolio allocation** and reusable strategy parameters
  (validated with Pydantic).
- Run a **buy-and-hold** or **periodically rebalanced** (monthly / quarterly /
  annually) simulation with transparent accounting.
- Calculate **portfolio value, total return, CAGR, annualised volatility and
  maximum drawdown**.
- Display an interactive **equity curve** and **drawdown** chart (Plotly).
- **Compare** the simulated strategy against a chosen benchmark asset.
- **Unit tests** for returns, drawdown, rate conversion and portfolio accounting.

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
# Clone, then from the project root:
uv sync --extra dev
```

`uv sync` creates a virtual environment in `.venv/` and installs the runtime and
development dependencies pinned in `pyproject.toml`.

## Running the app

```bash
uv run streamlit run app/Home.py
```

Streamlit opens the interface in your browser. Pick a data source in the
sidebar — the bundled **Stocks + Cash (1954+)** dataset (S&P 500 alongside a
cash account that compounds the federal funds rate, from `data/`), the
full-history **S&P 500 (daily)** series, or your own uploaded wide-format CSV —
then set the allocation weights, choose a rebalancing cadence, and run a
simulation. (The `data/` files are git-ignored and provided locally.)

> **Note on "Cash (Fed Funds)":** the federal funds rate is a short-term
> overnight rate. Compounding it models a risk-free cash / money-market account,
> **not** long-term bonds — it has no duration or price risk.

## Using your own data

Provide a CSV with a `date` column and one closing-price column per asset:

```csv
date,STOCKS,BONDS,GOLD
2019-01-02,100.0,100.0,100.0
2019-01-03,100.8,100.1,99.4
```

## Using the engine directly

```python
from portfolio_research_lab import (
    StrategyConfig,
    load_price_data,
    load_rate_series,
    rate_to_index,
    run_simulation,
)

# Stocks + a cash account that compounds the federal funds rate.
stocks = load_price_data("data/sp500daily.csv").rename(columns={"close": "S&P 500"})
cash = rate_to_index(load_rate_series("data/fed-funds-rate.csv")).rename("Cash (Fed Funds)")
prices = stocks.join(cash, how="left").ffill().dropna(how="any")  # trims to 1954+

config = StrategyConfig(
    name="60/40, rebalanced annually",
    initial_capital=10_000,
    allocations={"S&P 500": 0.6, "Cash (Fed Funds)": 0.4},
    rebalance_frequency="annually",  # None = buy & hold
)
result = run_simulation(prices, config)
print(result.metrics())
```

## Testing, linting and formatting

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Project structure

```
market-simulation-lab/
├── app/
│   └── Home.py                    # Streamlit UI (presentation layer only)
├── src/
│   └── portfolio_research_lab/
│       ├── __init__.py            # public API
│       ├── models.py              # Pydantic configuration models
│       ├── data.py                # CSV / rate loading + cleaning
│       ├── strategies.py          # strategy definitions (buy-and-hold)
│       ├── simulator.py           # portfolio accounting / engine
│       └── metrics.py             # returns, CAGR, volatility, drawdown
├── tests/                         # pytest unit tests
├── data/                          # local price CSVs (git-ignored)
├── pyproject.toml
├── README.md
└── .gitignore
```

## Scope and non-goals

To keep the foundation small and understandable, this project deliberately
avoids databases, authentication, cloud services, machine learning and
premature optimization. Transaction costs, taxes and leverage are **not**
modelled yet — the strategy interface leaves room to add them later.

## License

MIT

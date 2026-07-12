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
> It uses simplified models and synthetic sample data, and it is **not financial
> advice**. Nothing here is a recommendation to buy or sell any security. Do
> your own research and consult a qualified professional before investing.

## Features

The initial release is a small but working foundation:

- Load historical asset prices from wide-format **CSV files**.
- Configure an **initial portfolio allocation** and reusable strategy parameters
  (validated with Pydantic).
- Run a **buy-and-hold** portfolio simulation with transparent accounting.
- Calculate **portfolio value, total return, CAGR, annualised volatility and
  maximum drawdown**.
- Display an interactive **equity curve** and **drawdown** chart (Plotly).
- **Compare** the simulated strategy against a chosen benchmark asset.
- **Synthetic sample data** generated on first run so the app works immediately.
- **Unit tests** for returns, drawdown and portfolio accounting.

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

Streamlit opens the interface in your browser. On first run it generates a
synthetic sample dataset (`sample_data/sample_prices.csv`, which is git-ignored),
so you can configure an allocation and run a simulation right away. You can also
upload your own wide-format CSV from the sidebar.

## Using your own data

Provide a CSV with a `date` column and one closing-price column per asset:

```csv
date,STOCKS,BONDS,GOLD
2019-01-02,100.0,100.0,100.0
2019-01-03,100.8,100.1,99.4
```

To regenerate the bundled synthetic dataset:

```bash
uv run python sample_data/generate_sample_data.py
```

## Using the engine directly

```python
from portfolio_research_lab import load_price_data, StrategyConfig, run_simulation

prices = load_price_data("sample_data/sample_prices.csv")
config = StrategyConfig(
    name="60/40",
    initial_capital=10_000,
    allocations={"STOCKS": 0.6, "BONDS": 0.4},
    benchmark="BENCHMARK",
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
│       ├── data.py                # CSV loading + synthetic data generation
│       ├── strategies.py          # strategy definitions (buy-and-hold)
│       ├── simulator.py           # portfolio accounting / engine
│       └── metrics.py             # returns, CAGR, volatility, drawdown
├── tests/                         # pytest unit tests
├── sample_data/
│   └── generate_sample_data.py    # generates sample_prices.csv (git-ignored)
├── pyproject.toml
├── README.md
└── .gitignore
```

## Scope and non-goals

To keep the foundation small and understandable, this project deliberately
avoids databases, authentication, cloud services, machine learning and
premature optimization. Transaction costs, taxes, leverage, cash yield and
rebalancing are **not** modelled yet — the strategy interface leaves room to add
them later.

## License

MIT

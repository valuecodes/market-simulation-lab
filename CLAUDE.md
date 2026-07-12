# CLAUDE.md

## What this is

Portfolio Research Lab — a local-first Python app for backtesting investment
strategies.

## Commands

Tasks are defined as [poe](https://poethepoet.natn.io/) tasks in `pyproject.toml`.

```bash
uv sync --extra dev   # install deps (creates .venv/)
uv run poe dev        # run the Streamlit app
uv run poe lint       # ruff check .
uv run poe fmt        # ruff format .
uv run poe test       # pytest
uv run poe check      # lint + test
uv run poe            # list all tasks
```

## Conventions

- Python ≥ 3.12, managed with `uv`.
- ruff for lint + format (line length 100, double quotes); config in `pyproject.toml`.
- Run `uv run poe check` before committing.

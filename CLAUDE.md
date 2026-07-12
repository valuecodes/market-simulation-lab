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
uv run poe typecheck  # ty check
uv run poe test       # pytest
uv run poe cov        # pytest + coverage report
uv run poe check      # lint + typecheck + test
uv run poe            # list all tasks
```

## Conventions

- Python ≥ 3.12, managed with `uv`.
- ruff for lint + format (line length 100, double quotes); `ty` for type checking.
  All config lives in `pyproject.toml`. Keep the code fully type-annotated.
- Run `uv run poe check` before committing. CI (`.github/workflows/ci.yml`) runs
  the same gates on push and PRs.

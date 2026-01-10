# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core ingestion, metrics, API helpers, and the `bsw` CLI entry point.
- `dashboard/`: Dash app, callbacks, layouts, and static assets in `dashboard/assets/`.
- `tests/`: pytest suite (unit tests plus Dash E2E checks).
- `data/`, `resources/`, `tmp/`: local data artifacts and working files (typically untracked).
- `DDL.sql`/`SEED.sql`: database schema and seed data for DuckDB.

## Build, Test, and Development Commands
- `uv sync --dev`: install dependencies from `pyproject.toml`/`uv.lock`.
- `uv run bsw init-db --db data/db/music.db`: initialize a DuckDB file with `DDL.sql`.
- `uv run bsw ingest-history --user-id <uuid> --history-dir data/<user>/listening_history`: ingest Spotify JSON history into `fact_plays`.
- `uv run python dashboard/application.py`: run the Dash app locally.
- `pytest`: run all tests in `tests/` (`pytest -k <pattern>` for a subset).

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation and max line length of 100.
- Ruff is the linter/formatter (`ruff check --fix .`, `ruff format .`); prefer double quotes per `ruff.toml`.
- Module names are lowercase with underscores; CLI commands are kebab-case (e.g., `ingest-history`).

## Testing Guidelines
- Pytest is configured via `pytest.ini` with `tests/` as the root.
- Dash E2E tests use `dash[testing]` and may be skipped in CI (`CI` env var).
- Test files follow `tests/test_*.py`; keep fixtures in `tests/conftest.py`.

## Commit & Pull Request Guidelines
- Commit subjects are short, descriptive, and imperative (e.g., “Update album calculation”).
- PRs should include: a brief summary, key changes, and test evidence.
- For dashboard/UI changes, include a screenshot or GIF.
- Link related issues or tasks when applicable.

## Configuration Tips
- Store API keys in a local `.env` (never commit secrets).
- DuckDB lives at `data/db/music.db` by default; keep large data out of git.

# Repository Guidelines

## Project Structure & Modules
- `src/`: Core Python code (CLI in `src/cli.py`, data/API in `src/api/`, metrics in `src/metrics/`).
- `dashboard/`: Dash app (`application.py`, `callbacks/`, `components/`, `layouts/`, `assets/`).
- `tests/`: Pytest suites (unit + Dash integration).
- `data/`: Local datasets and cache (ignored); DB at `data/db/music.db`.
- Root config: `pyproject.toml`, `ruff.toml`, `pytest.ini`, `DDL.sql`, `SEED.sql`.

## Build, Test, and Dev Commands
- Install deps (uv): `uv sync`
- Run tests: `uv run pytest -ra` (CI: `-k "not test_app_starts"`).
- Lint/format (ruff): `uv run ruff check --fix .` and `uv run ruff format .`
- Start dashboard: `uv run python dashboard/application.py`
- CLI (ingest): `uv run bsw ingest-history --user-id <id> --db data/db/music.db --history-dir data/<id>/listening_history --apply-ddl`

## Coding Style & Naming
- Python 3.10+, 4‑space indent, line length 100, double quotes.
- Use type hints and snake_case for functions/modules; PascalCase for classes.
- Imports managed by Ruff (isort); keep first‑party as `src`, `dashboard`.
- Run Ruff before committing; pre-commit is configured to run Ruff and tests.

## Testing Guidelines
- Framework: Pytest (+ `dash[testing]` for UI). Tests live in `tests/` as `test_*.py`.
- E2E UI test `test_app_starts` is skipped in CI (`CI=1`). Locally, ensure Chrome is installed; driver is handled by `webdriver-manager`.
- Target small, deterministic fixtures; prefer in‑memory DuckDB for SQL paths.
- Quick filters: `uv run pytest tests/test_metrics.py::test_get_top_albums`.

## Commit & PR Guidelines
- Commits: imperative, concise (e.g., "Add genre L0 mapping", "Fix pandas warning"). Group related changes.
- PRs: clear description, linked issues, repro steps; include screenshots/gifs for dashboard/UI changes and notes on data migrations (DDL/SEED). Ensure tests pass and style checks are clean.

## Security & Configuration
- Required env: `.env` with `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, optionally `DATA_DIR`.
- Never commit credentials or large data; `data/` and `.env` are git‑ignored.
- For multiple worktrees, bootstrap shared paths: `scripts/bootstrap_worktree.sh --from <primary>`.

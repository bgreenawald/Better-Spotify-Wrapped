# Repository Guidelines

Architecture, module overview, CLI commands, environment variables, testing and
code-style conventions live in [`CLAUDE.md`](CLAUDE.md); user-facing setup and the
pipeline order are in [`README.md`](README.md). Run `uv run slopify --help` for the
authoritative command list. This file only holds agent-workflow rules; do not copy
command lists here.

## Commit & Pull Request Guidelines
- Commit subjects are short, descriptive, and imperative (e.g., "Update album calculation").
- PRs should include: a brief summary, key changes, and test evidence.
- For dashboard/UI changes, include a screenshot or GIF.
- Link related issues or tasks when applicable.

## Working Rules
- Never commit secrets: API keys belong in a local `.env` (see `.env.example`).
- Keep large data (`data/`, `resources/`, `tmp/`) out of git.
- Dash E2E tests may be skipped in CI (`CI` env var); run `uv run python -m pytest tests/` locally before opening a PR.

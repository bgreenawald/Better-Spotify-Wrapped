# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Better Spotify Wrapped is a tool for analyzing Spotify listening history with more detailed insights than the standard Spotify Wrapped. The project consists of:

1. **Data ingestion pipeline**: Loads Spotify listening history JSON exports into a normalized DuckDB database
2. **Spotify API integration**: Fetches track metadata, artist info, album details, and genres with local caching
3. **Metrics calculation**: Computes listening trends, social listening patterns, and genre analysis
4. **Interactive dashboard**: Dash web application for visualizing listening patterns over time

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Run with uv
uv run <command>
```

### Testing
```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test file
uv run python -m pytest tests/test_metrics.py

# Run specific test function
uv run python -m pytest tests/test_metrics.py::test_function_name

# Run with verbose output
uv run python -m pytest tests/ -v
```

### Linting and Formatting
```bash
# Run ruff linter (with autofix)
uv run ruff check --fix

# Run ruff formatter
uv run ruff format

# Pre-commit hooks (linting + pytest, excluding test_app_starts)
pre-commit run --all-files
```

### CLI Commands

The project provides a `bsw` CLI tool (defined in src/cli.py):

```bash
# Initialize database schema
uv run bsw init-db --db data/db/music.db --ddl DDL.sql

# Ingest listening history from Spotify export JSON files
uv run bsw ingest-history --user-id <user-id> --history-dir data/<user>/listening_history --db data/db/music.db --apply-ddl

# Populate track metadata (ISRC, duration, explicit flag)
uv run bsw ingest-track-metadata --db data/db/music.db

# Populate album information
uv run bsw ingest-track-albums --db data/db/music.db

# Populate artist information and bridge tables
uv run bsw ingest-track-artists --db data/db/music.db

# Load artist genre evidence from cache
uv run bsw ingest-artist-genres --db data/db/music.db

# Link artists to canonical genres
uv run bsw link-artist-genres --db data/db/music.db

# Populate tracks from cached album data
uv run bsw ingest-tracks-from-albums --db data/db/music.db
```

### Dashboard

```bash
# Run dashboard locally
uv run python dashboard/application.py
```

The dashboard runs on http://localhost:8050 by default.

## Architecture

### Database Schema (DuckDB)

The project uses a normalized DuckDB schema defined in `DDL.sql`:

- **Dimension tables**: `dim_users`, `dim_tracks`, `dim_albums`, `dim_artists`, `dim_genres`, `dim_moods`
- **Fact table**: `fact_plays` (listening history events)
- **Bridge tables**: `bridge_track_artists` (handles features, remixers with role field)
- **Genre/Mood mapping**: `tag_evidence` (raw tags from APIs), `map_genre`/`map_mood` (mappings to canonical IDs)
- **Relationship tables**: `artist_genres`, `track_genres`, `track_moods`, `genre_hierarchy`
- **Aggregation tables**: `agg_user_genre_daily`, `agg_user_genre_monthly`, `agg_user_mood_daily`, `agg_user_mood_monthly`
- **Views**: `v_primary_artist_per_track`, `v_plays_enriched` (denormalized plays with track/album/artist)

### Data Flow

1. **Ingestion** (`src/db_ingest.py`): Loads raw Spotify JSON exports into `fact_plays` and minimal `dim_tracks`/`dim_users` records
2. **Enrichment** (`src/api/api.py`): Uses Spotify API (via spotipy) to fetch metadata; results cached in `data/api/cache/`
3. **Genre taxonomy** (`src/load_genre_taxonomy.py`): Loads hierarchical genre structure with parent-child relationships
4. **Metrics** (`src/metrics/`): Computes trends, social listening patterns, and statistics
5. **Dashboard** (`dashboard/`): Dash application with callbacks, components, and layouts for visualization

### Key Modules

**src/cli.py**: CLI entry point with Click commands for database operations

**src/api/api.py**:
- `SpotifyDataCollector`: Fetches and caches track/artist/album data from Spotify API
- Functions like `populate_track_metadata()`, `populate_track_albums()`, `populate_track_artists()` populate database tables from cached API responses
- Thread-safe caching with file locks to avoid concurrent write issues

**src/db_ingest.py**:
- `load_history_into_fact_plays()`: Main function to load Spotify JSON exports into database
- Returns `IngestResult` with counts of inserted/deduped plays and tracks

**src/preprocessing.py**:
- Legacy data preparation utilities (now partially superseded by DB-backed approach)
- `add_api_data()`: Merges API data with listening history DataFrames

**src/metrics/metrics.py**:
- Core metrics calculations for listening patterns
- Functions operate on pandas DataFrames loaded from database

**src/metrics/social.py**:
- Social listening metrics (e.g., group listening patterns)

**src/metrics/trends.py**:
- Trend analysis over time (monthly/daily aggregations)

**dashboard/application.py**:
- `create_app()`: Initializes Dash app, loads data from DuckDB via `_load_base_dataframe()`
- Uses `dashboard.conn.get_db_connection()` to connect to database

**dashboard/conn.py**:
- `get_db_connection()`: Returns DuckDB connection (read-only by default)
- Database path from `MUSIC_DB` environment variable (default: `data/db/music.db`)

**dashboard/callbacks/callbacks.py**:
- Dash callbacks for interactive filtering and chart updates

**dashboard/components/**:
- UI components (filters, graphs, stats cards)

**dashboard/layouts/layouts.py**:
- Page layouts and tab structures

### Environment Variables

The project uses a `.env` file with the following keys:

- `SPOTIFY_CLIENT_ID`: Spotify API client ID (required for API calls)
- `SPOTIFY_CLIENT_SECRET`: Spotify API client secret (required for API calls)
- `DATA_DIR`: Base directory for user data (e.g., `data/ben.greenawald`)
- `MUSIC_DB`: Path to DuckDB database (default: `data/db/music.db`)
- `SPOTIFY_API_CACHE_DIR`: Override for API cache directory (default: `data/api/cache`)
- `ENVIRONMENT`: Set to "local", "dev", "development", or "test" to override env vars from .env file

### API Caching

Spotify API responses are cached in `data/api/cache/` organized by resource type:
- `tracks/`: Track metadata JSON files (keyed by track ID)
- `artists/`: Artist metadata JSON files (keyed by artist ID)
- `albums/`: Album metadata JSON files (keyed by album ID)

Caching uses file locks to prevent concurrent writes and reduce API calls.

### Code Style

- Python 3.10+ (uses modern type hints like `str | None`)
- Ruff for linting and formatting (config in `ruff.toml`)
- Line length: 100 characters
- Double quotes for strings
- Type hints required (uses `from __future__ import annotations`)
- Pre-commit hooks run ruff and pytest (excluding `test_app_starts`)

### Testing Conventions

- Tests in `tests/` directory
- Use pytest for all tests
- `conftest.py` provides shared fixtures
- Test files: `test_*.py` pattern
- Pre-commit hook runs tests but skips `test_app_starts` (slow integration test)

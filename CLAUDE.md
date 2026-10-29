# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slopify Wrapped is a tool for analyzing Spotify listening history with more detailed insights than the standard Spotify Wrapped. The project consists of:

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

The project provides a `slopify` CLI tool (defined in src/cli.py):

```bash
# Initialize database schema
uv run slopify init-db --db data/db/music.db --ddl DDL.sql

# Ingest listening history from Spotify export JSON files
uv run slopify ingest-history --user-id <user-id> --history-dir data/<user>/listening_history --db data/db/music.db --apply-ddl

# Populate track metadata (ISRC, duration, explicit flag)
uv run slopify ingest-track-metadata --db data/db/music.db

# Populate album information
uv run slopify ingest-track-albums --db data/db/music.db

# Populate artist information and bridge tables
uv run slopify ingest-track-artists --db data/db/music.db

# Fetch artist JSONs from Spotify into the local cache
# (must run after ingest-track-artists and before ingest-artist-genres)
uv run slopify fetch-artist-cache --db data/db/music.db

# Fetch album JSONs into the local cache
# (must run after ingest-track-albums and before ingest-tracks-from-albums)
uv run slopify fetch-album-cache --db data/db/music.db

# Load artist genre evidence from cache
uv run slopify ingest-artist-genres --db data/db/music.db

# Link artists to canonical genres
uv run slopify link-artist-genres --db data/db/music.db

# Populate tracks from cached album data
uv run slopify ingest-tracks-from-albums --db data/db/music.db

# Narrower backfills of dim_tracks columns (see `uv run slopify --help` for all 11 commands)
uv run slopify ingest-isrcs --db data/db/music.db
uv run slopify ingest-duration-explicit --db data/db/music.db
```

```bash
# Report what is populated and what is missing; exits non-zero if
# anything needs attention, so it can gate a pipeline
uv run slopify status --db data/db/music.db
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

- **Dimension tables**: `dim_users`, `dim_tracks`, `dim_albums`, `dim_artists`, `dim_genres`
- **Fact table**: `fact_plays` (listening history events)
- **Bridge tables**: `bridge_track_artists` (handles features, remixers with role field)
- **Genre mapping**: `tag_evidence` (raw tags from APIs), `map_genre` (mapping to canonical IDs)
- **Relationship tables**: `artist_genres`, `genre_hierarchy`
- **Views**: `v_primary_artist_per_track`, `v_plays_enriched` (denormalized plays with track/album/artist)

That is the full set of 13 objects. Genre data comes from `artist_genres` only. There are no mood, `track_genres`, or pre-aggregated rollup tables (removed: no data source or writer/reader; aggregate from `fact_plays` directly).

### Data Flow

1. **Ingestion** (`src/db_ingest.py`): Loads raw Spotify JSON exports into `fact_plays` and minimal `dim_tracks`/`dim_users` records
2. **Enrichment** (`src/api/api.py`): Uses Spotify API (via spotipy) to fetch metadata; results cached in `data/api/cache/`. Order matters: `ingest-track-artists` -> `fetch-artist-cache` -> `ingest-artist-genres` -> `link-artist-genres`
3. **Genre taxonomy** (`src/load_genre_taxonomy.py`): Loads hierarchical genre structure with parent-child relationships
4. **Metrics** (`src/metrics/`): Computes trends, social listening patterns, and statistics
5. **Dashboard** (`dashboard/`): Dash application with callbacks, components, and layouts for visualization

### Key Modules

**src/cli.py**: CLI entry point with Click commands for database operations

**src/api/api.py**:
- `SpotifyDataCollector`: Fetches and caches track/artist/album data from Spotify API
- Functions like `populate_track_metadata()`, `populate_track_albums()`, `populate_track_artists()` populate database tables from cached API responses
- Caching: in-process `threading.Lock` only (no file locking); cache writes are atomic (temp file + `os.replace`); unreadable/truncated cache files count as a miss and are refetched; IDs Spotify cannot resolve go in a negative cache (`_dead/`) and are not re-requested

**src/link_artist_genres.py**: Links artists to canonical genres (backs `link-artist-genres`)

**src/load_genre_taxonomy.py**: Loads the hierarchical genre taxonomy into `dim_genres`/`genre_hierarchy`

**src/io.py**: `load_spotify_history()` reads raw export JSON into a DataFrame (legacy path)

**src/wrapped.py**: Standalone script that exports top played tracks per year to Excel (legacy)

**llm/**: Offline scripts that generate the genre subgenre taxonomy via an LLM (`get_subgenre_taxonomy.py`)

**scripts/bootstrap_worktree.sh**: Symlinks shared untracked paths (data, `.env`, etc.) into a new git worktree

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
- `DATA_DIR`: Optional, legacy only. Base directory for user data (e.g., `data/ben.greenawald`), read by `src/wrapped.py` and the legacy export helpers. The `ingest-*` commands do not need it.
- `MUSIC_DB`: Path to DuckDB database (default: `data/db/music.db`)
- `SPOTIFY_API_CACHE_DIR`: Override for API cache directory (default: `data/api/cache`)
- `DISPLAY_TIMEZONE`: IANA zone used to bucket plays by day/month (default: the
  machine's local zone; `UTC` groups by UTC days). `fact_plays.played_at` is
  stored as naive UTC and converted at query time in `get_filtered_plays` and
  `social.py`, so all tabs share one basis.
- `ENVIRONMENT`: Dashboard only (`dashboard/application.py`, `dashboard/conn.py`): set to "local", "dev", "development", or "test" to let `.env` override existing env vars. `src/api/api.py` and `src/api/utils.py` call `load_dotenv(override=True)` unconditionally, so CLI runs always let `.env` clobber shell env vars

### API Caching

Spotify API responses are cached in `data/api/cache/` organized by resource type:
- `tracks/`: Track metadata JSON files (keyed by track ID)
- `artists/`: Artist metadata JSON files (keyed by artist ID)
- `albums/`: Album metadata JSON files (keyed by album ID)

Each category also has a `_dead/` subdirectory of marker files for IDs Spotify cannot resolve (negative cache). Writes are atomic (temp file + `os.replace`) and corrupt files are refetched; there is no file locking.

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

# Slopify Wrapped

A tool for analyzing Spotify listening history with more detailed insights than the standard Spotify Wrapped.

## Screenshots

<!--
  PLACEHOLDER: add 3-4 dashboard screenshots here (e.g. the overview tab, top
  artists/tracks, genre trends, social listening). Save the images under
  docs/images/ and reference them like:

  ![Overview](docs/images/overview.png)

  Remove this comment once real images are added. Do not link images that do
  not exist.
-->

_Screenshots coming soon._

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer
(`requires-python = ">=3.10"` in `pyproject.toml`). Local development and the
`.python-version` file use Python 3.13.

```bash
git clone <this-repo-url>
cd better_spotify_wrapped
uv sync          # creates .venv and installs runtime + dev dependencies
cp .env.example .env   # then fill in your Spotify credentials
uv run slopify --help
```

## Configuration

Copy `.env.example` to `.env`; it documents every variable. Only the two Spotify
keys are required (and only for the commands that call the Spotify API). Get
credentials by creating an app at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

| Variable | Required | Purpose |
|----------|----------|---------|
| `SPOTIFY_CLIENT_ID` | Yes | Spotify API client ID |
| `SPOTIFY_CLIENT_SECRET` | Yes | Spotify API client secret |
| `MUSIC_DB` | No | DuckDB path (default `data/db/music.db`) |
| `SPOTIFY_API_CACHE_DIR` | No | API cache directory (default `data/api/cache`) |
| `SPOTIFY_API_RATE_DELAY` | No | Seconds between API requests (default `1.0`) |
| `DISPLAY_TIMEZONE` | No | IANA zone used to bucket plays into days/months in the dashboard (default: machine local zone; `UTC` groups by UTC days) |
| `DATA_DIR` | No | Legacy only: used by the old whole-export helpers, not needed for the pipeline below |
| `ENVIRONMENT` | No | `local`/`dev`/`development`/`test` lets `.env` override existing environment variables |
| `LLM_PROVIDER`, `GEMINI_MODEL`, `OPENAI_MODEL`, `GOOGLE_API_KEY`, `OPENAI_API_KEY` | No | Only for the genre taxonomy scripts in `llm/` |

## Loading a New Year's Data

Follow these steps to load a new year's Spotify listening history data into the database.

### Prerequisites

1. **Request your Spotify data export** from [Spotify Privacy Settings](https://www.spotify.com/account/privacy/). Select "Extended streaming history" to get the detailed JSON files needed.

2. **Wait for Spotify to send you the data** (can take up to 30 days, but usually arrives within a week).

3. **Extract the data** and locate the `Streaming_History_Audio_*.json` files from your export.

### Step 1: Organize Your Data

Place your Spotify JSON export files in a user-specific directory:

```
data/<user-id>/listening_history/
```

For example:
```
data/john.doe/listening_history/
├── Streaming_History_Audio_2024-2025_0.json
├── Streaming_History_Audio_2025_1.json
└── ...
```

The JSON files should follow Spotify's export format with fields like:
- `ts` - Timestamp of play
- `spotify_track_uri` - Spotify track identifier
- `ms_played` - Duration played in milliseconds
- `master_metadata_track_name` - Track name
- `master_metadata_album_artist_name` - Artist name

### Step 2: Initialize the Database (First Time Only)

If this is your first time setting up the database:

```bash
uv run slopify init-db --db data/db/music.db --ddl DDL.sql
```

### Step 3: Ingest Listening History

Load the JSON files into the database:

```bash
uv run slopify ingest-history \
  --user-id <your-user-id> \
  --history-dir data/<your-user-id>/listening_history \
  --db data/db/music.db
```

If you need to initialize the schema and ingest in one command:

```bash
uv run slopify ingest-history \
  --user-id <your-user-id> \
  --history-dir data/<your-user-id>/listening_history \
  --db data/db/music.db \
  --apply-ddl
```

**Note:** The ingestion is idempotent - duplicate plays (same user, track, and timestamp) are automatically skipped. You can safely re-run this command if you get updated exports.

### Step 4: Enrich Track Metadata

After ingesting the basic play history, enrich the tracks with metadata from the Spotify API. The `fetch-*-cache` commands download API responses into the local cache; the `ingest-*` commands read from that cache, so each fetch must run **before** the ingest that depends on it.

```bash
# Populate ISRC, duration, and explicit flag
uv run slopify ingest-track-metadata --db data/db/music.db

# Populate album information
uv run slopify ingest-track-albums --db data/db/music.db

# Fetch album JSONs into the cache, then load every track on those albums
uv run slopify fetch-album-cache --db data/db/music.db
uv run slopify ingest-tracks-from-albums --db data/db/music.db

# Populate artist information
uv run slopify ingest-track-artists --db data/db/music.db
```

`ingest-tracks-from-albums` lets you see all tracks from albums you've listened to, not just the ones you played.

### Step 5: Load Genre Information

```bash
# Fetch artist JSONs into the cache
uv run slopify fetch-artist-cache --db data/db/music.db

# Load artist genre evidence from the cached API responses
uv run slopify ingest-artist-genres --db data/db/music.db

# Link artists to canonical genres
uv run slopify link-artist-genres --db data/db/music.db
```

### Step 6: Verify

```bash
uv run slopify status --db data/db/music.db
```

See [Verifying your setup](#verifying-your-setup).

### Complete Example

Here's a complete workflow for a new user `jane.smith`:

```bash
# 1. Create directory and copy Spotify export files
mkdir -p data/jane.smith/listening_history
cp ~/Downloads/my_spotify_data/Streaming_History_Audio_*.json data/jane.smith/listening_history/

# 2. Initialize DB and ingest history
uv run slopify init-db --db data/db/music.db --ddl DDL.sql
uv run slopify ingest-history \
  --user-id jane.smith \
  --history-dir data/jane.smith/listening_history \
  --db data/db/music.db

# 3. Enrich tracks, albums and artists (fetch-* steps fill the caches the next step reads)
uv run slopify ingest-track-metadata --db data/db/music.db && \
uv run slopify ingest-track-albums --db data/db/music.db && \
uv run slopify fetch-album-cache --db data/db/music.db && \
uv run slopify ingest-tracks-from-albums --db data/db/music.db && \
uv run slopify ingest-track-artists --db data/db/music.db

# 4. Load genre information
uv run slopify fetch-artist-cache --db data/db/music.db && \
uv run slopify ingest-artist-genres --db data/db/music.db && \
uv run slopify link-artist-genres --db data/db/music.db

# 5. Check that everything is populated
uv run slopify status --db data/db/music.db
```

### Verifying your setup

`slopify status` prints one line per check with its count and an `[ok  ]` or failing marker: schema objects, users, plays, tracks, date range, orphan plays, tracks missing duration/ISRC/album/primary artist, artists, genre taxonomy, and genre evidence. It exits non-zero when something needs attention, so it can also gate a script or CI step:

```bash
uv run slopify status --db data/db/music.db && echo "database looks complete"
```

If a check fails, re-run the pipeline step that populates it (see the order above). Every step is safe to re-run.

### CLI Reference

Run `uv run slopify <command> --help` for full options. `--db` is accepted by every command (default `data/db/music.db`).

| Command | Description | `--limit` | `--cache-dir` |
|---------|-------------|:---------:|:-------------:|
| `slopify init-db` | Initialize database schema from DDL.sql (`--ddl`) | no | no |
| `slopify ingest-history` | Load listening history JSON into fact_plays (`--user-id`, `--history-dir`, `--apply-ddl`, `--ddl`) | no | no |
| `slopify ingest-track-metadata` | Populate ISRC, duration_ms, explicit in one batch | yes | yes |
| `slopify ingest-track-albums` | Populate dim_tracks.album_id and dim_albums | yes | yes |
| `slopify fetch-album-cache` | Fetch album JSONs from Spotify into the cache (`--workers`) | yes | yes |
| `slopify ingest-tracks-from-albums` | Load tracks from cached album JSONs into dim_tracks | yes | yes |
| `slopify ingest-track-artists` | Populate dim_artists and bridge_track_artists | yes | yes |
| `slopify fetch-artist-cache` | Fetch artist JSONs from Spotify into the cache (`--workers`) | yes | yes |
| `slopify ingest-artist-genres` | Load artist genre evidence from cached artist JSONs | yes | yes |
| `slopify link-artist-genres` | Link artists to canonical genres | yes | yes |
| `slopify status` | Report what is populated and what is missing; non-zero exit if attention needed | no | no |
| `slopify ingest-isrcs` | Fill missing dim_tracks.track_isrc (cache + API) | yes | no |
| `slopify ingest-duration-explicit` | Fill missing duration_ms and explicit | yes | yes |

`ingest-track-metadata` already covers what `ingest-isrcs` and `ingest-duration-explicit` do; those two are for targeted backfills. Global option: `-v`/`-vv` for more log output.

## Running the Dashboard

```bash
uv run python dashboard/application.py
```

The dashboard runs at http://localhost:8050 by default.

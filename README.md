# Better Spotify Wrapped

A tool for analyzing Spotify listening history with more detailed insights than the standard Spotify Wrapped.

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
uv run bsw init-db --db data/db/music.db --ddl DDL.sql
```

### Step 3: Ingest Listening History

Load the JSON files into the database:

```bash
uv run bsw ingest-history \
  --user-id <your-user-id> \
  --history-dir data/<your-user-id>/listening_history \
  --db data/db/music.db
```

If you need to initialize the schema and ingest in one command:

```bash
uv run bsw ingest-history \
  --user-id <your-user-id> \
  --history-dir data/<your-user-id>/listening_history \
  --db data/db/music.db \
  --apply-ddl
```

**Note:** The ingestion is idempotent - duplicate plays (same user, track, and timestamp) are automatically skipped. You can safely re-run this command if you get updated exports.

### Step 4: Enrich Track Metadata

After ingesting the basic play history, enrich the tracks with metadata from the Spotify API:

```bash
# Populate ISRC, duration, and explicit flag
uv run bsw ingest-track-metadata --db data/db/music.db

# Populate album information
uv run bsw ingest-track-albums --db data/db/music.db

# Populate artist information
uv run bsw ingest-track-artists --db data/db/music.db
```

### Step 5: Load Genre Information

```bash
# Load artist genre evidence from cached API responses
uv run bsw ingest-artist-genres --db data/db/music.db

# Link artists to canonical genres
uv run bsw link-artist-genres --db data/db/music.db
```

### Step 6: (Optional) Load Additional Tracks from Albums

If you want to see all tracks from albums you've listened to (not just the ones you played):

```bash
uv run bsw ingest-tracks-from-albums --db data/db/music.db
```

### Complete Example

Here's a complete workflow for a new user `jane.smith`:

```bash
# 1. Create directory and copy Spotify export files
mkdir -p data/jane.smith/listening_history
cp ~/Downloads/my_spotify_data/Streaming_History_Audio_*.json data/jane.smith/listening_history/

# 2. Initialize DB and ingest history
uv run bsw ingest-history \
  --user-id jane.smith \
  --history-dir data/jane.smith/listening_history \
  --db data/db/music.db \
  --apply-ddl

# 3. Enrich with Spotify API metadata
uv run bsw ingest-track-metadata --db data/db/music.db && \
uv run bsw ingest-track-albums --db data/db/music.db && \
uv run bsw ingest-track-artists --db data/db/music.db

# 4. Load genre information
uv run bsw ingest-artist-genres --db data/db/music.db && \
uv run bsw link-artist-genres --db data/db/music.db
```

### Environment Setup for API Enrichment

To fetch metadata from the Spotify API, create a `.env` file with your Spotify API credentials:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

Get credentials by creating an app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

### CLI Reference

| Command | Description |
|---------|-------------|
| `bsw init-db` | Initialize database schema from DDL.sql |
| `bsw ingest-history` | Load listening history JSON into fact_plays |
| `bsw ingest-track-metadata` | Populate ISRC, duration_ms, explicit |
| `bsw ingest-track-albums` | Populate album_id and dim_albums |
| `bsw ingest-track-artists` | Populate dim_artists and bridge_track_artists |
| `bsw ingest-artist-genres` | Load artist genre evidence from cache |
| `bsw link-artist-genres` | Link artists to canonical genres |
| `bsw ingest-tracks-from-albums` | Load tracks from cached album data |

All commands support `--limit` to cap the number of records processed and `--cache-dir` to override the API cache location.

## Running the Dashboard

```bash
uv run python dashboard/application.py
```

The dashboard runs at http://localhost:8050 by default.

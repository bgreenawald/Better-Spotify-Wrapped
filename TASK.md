
# TASK.md

## Overview

We are migrating an existing prototype Spotify listening analytics app into a more robust system. The original prototype was built in Python (using Dash) and focused on user listening data and simple aggregations. The new design introduces richer **genre** and **mood** classification using multiple metadata sources, a canonical taxonomy, and a normalized schema for reproducible analytics.

The agent’s role is to help **implement and migrate the prototype** to this new architecture.

---

## Key Goals

1. **Multi-user support** — system should handle multiple users’ listening histories.
2. **Rich genre + mood enrichment**:

   * Collect raw tags from multiple sources (MusicBrainz, Last.fm, Discogs, TheAudioDB).
   * Map them to canonical vocabularies (genres, moods).
   * Compute normalized per-track distributions (summing to 1).
   * Store both provenance (`tag_evidence`) and normalized scores (`track_genres`, `track_moods`).
3. **Rollups for analytics**:

   * Daily and monthly aggregations of genres and moods per user.
   * Enable queries like “top genres/moods over last 90 days” or “genre drift over time.”
4. **Advanced queries**:

   * Support album-level statistics (e.g. *median number of track plays per album*).
   * Include tracks with zero plays when computing album medians.
5. **Local-first design**:

   * No OAuth server or online auth flows required (Spotify data comes from extended streaming history JSON exports).
   * API tokens for external sources can be stored locally in `.env` (if needed).

---

## Data Flow (Track → Genres/Moods)

1. **Ingest** Spotify history (`fact_plays`) with track IDs, play timestamps, durations.
2. **Resolve IDs** (via ISRC → MusicBrainz MBID; fallback fuzzy search).
3. **Collect raw evidence**:

   * MusicBrainz: recording/release/artist genres + tags.
   * Discogs: release/artist genres & styles.
   * Last.fm: `track.getTopTags` + `artist.getTopTags` (community tags).
   * TheAudioDB: explicit `Mood` and `Style` fields.
4. **Store evidence** in `tag_evidence`.
5. **Map evidence → canonical vocabularies** via `map_genre` and `map_mood`.
6. **Compute effective weights**:
   `effective_weight = raw_weight × mapping_confidence × source_prior`
7. **Aggregate per track**, normalize to probabilities → `track_genres` and `track_moods`.
8. **Roll up** to per-user daily/monthly aggregates.

---

## Database Schema (DDL)

### Core Dimensions

* `dim_users` — users
* `dim_tracks`, `dim_albums`, `dim_artists` — identity tables
* `bridge_track_artists` — many-to-many track ↔ artist
* `fact_plays` — user listening history (track-level)

### Canonical Vocabularies

* `dim_genres` — canonical genre list
* `dim_moods` — canonical mood list

### Evidence & Mapping

* `tag_evidence` — raw tags from external sources (with provenance, weights)
* `map_genre` — source-specific tag → genre mapping (with confidence)
* `map_mood` — source-specific tag → mood mapping (with confidence)

### Normalized Per-track

* `track_genres` — normalized genre distribution per track
* `track_moods` — normalized mood distribution per track

### Aggregates

* `agg_user_genre_daily` / `agg_user_genre_monthly`
* `agg_user_mood_daily` / `agg_user_mood_monthly`

(See `music_schema_proposed.sql` for full DDL.)

---

## Key Query Example (album medians)

For each user and year:

* Identify albums they listened to.
* Gather **all tracks** from those albums (even if not played).
* Count plays per track (0 if none).
* Compute **median play count** across all tracks in album.

---

## Frontend Analytics

* Prototype built in **Dash**.
* Migration path options:

  * **Streamlit** for faster Python-native iteration.
  * Or **Next.js + FastAPI** for a more polished product UI.
* Analytics features to include:

  * Genre/mood distributions over time.
  * Heatmaps (genre × week, mood × hour).
  * Album-level medians.
  * Diversity metrics (entropy).

---

## Deliverables for Migration

1. Implement the new schema in the DB (SQLite/DuckDB/Postgres-compatible).
2. Build ETL scripts:

   * Ingest Spotify exports.
   * Enrich with external metadata.
   * Normalize into `track_genres` / `track_moods`.
   * Compute rollups.
3. Adapt frontend to read from new rollups and display:

   * Genre/mood trends, heatmaps, and diversity metrics.
   * Album-level median play counts.
4. Ensure reproducibility:

   * Version taxonomy files (e.g., `genres_taxonomy.yaml` + `moods_taxonomy.yaml`).
   * Store `method_version` in normalized tables.


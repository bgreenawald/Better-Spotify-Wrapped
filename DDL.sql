CREATE TABLE dim_users (
  user_id         TEXT PRIMARY KEY,      -- internal UUID
  display_name    TEXT,
  country         TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dim_tracks (
  track_id         TEXT PRIMARY KEY, -- Spotify ID
  track_mbid TEXT,
  track_isrc       TEXT,
  track_name       TEXT NOT NULL,
  album_id         TEXT,                   -- Spotify ID (album)
  duration_ms      INT,
  explicit         BOOLEAN,
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dim_albums (
  album_id         TEXT PRIMARY KEY,       -- Spotify ID
  album_name       TEXT NOT NULL,
  release_year     INT,
  label            TEXT
);

CREATE TABLE dim_artists (
  artist_id        TEXT PRIMARY KEY,       -- Spotify ID
  artist_name      TEXT NOT NULL
);

-- Bridge between tracks and artists (handles features, remixers, etc.)
CREATE TABLE bridge_track_artists (
  track_id   TEXT NOT NULL REFERENCES dim_tracks(track_id),
  artist_id  TEXT NOT NULL REFERENCES dim_artists(artist_id),
  role       TEXT NOT NULL,                -- 'primary' | 'feature' | 'remixer'
  PRIMARY KEY (track_id, artist_id, role)
);

CREATE TABLE fact_plays (
  play_id       TEXT PRIMARY KEY,         -- UUID
  user_id       TEXT NOT NULL REFERENCES dim_users(user_id),
  track_id      TEXT NOT NULL REFERENCES dim_tracks(track_id),
  played_at     TIMESTAMP NOT NULL,
  context_type  TEXT,                     -- playlist | album | radio | unknown
  context_id    TEXT,
  device_type   TEXT,
  duration_ms   INT,                      -- actual listened ms
  conn_country  TEXT,                     -- from export: connection country
  ip_addr       TEXT,                     -- from export: IP address
  reason_start  TEXT,                     -- from export: reason_start
  reason_end    TEXT,                     -- from export: reason_end
  shuffle       BOOLEAN,                  -- from export
  skipped       BOOLEAN,                  -- from export
  offline       BOOLEAN,                  -- from export
  offline_timestamp TIMESTAMP,            -- from export
  incognito_mode BOOLEAN,                 -- from export
  UNIQUE(user_id, track_id, played_at)    -- dedup key
);

-- DuckDB: prefer an explicit sequence to auto-increment
CREATE SEQUENCE IF NOT EXISTS seq_dim_genres START 1;

-- Canonical genres (slug-based external key; hierarchy modeled separately)
CREATE TABLE dim_genres (
  genre_id   INTEGER PRIMARY KEY DEFAULT nextval('seq_dim_genres'),
  slug       TEXT UNIQUE NOT NULL,   -- stable external key
  name       TEXT NOT NULL,          -- display name
  level      INTEGER,                -- 0 = parent, 1 = child, etc.
  active     BOOLEAN DEFAULT TRUE
);

CREATE TABLE dim_moods (
  mood_id        INTEGER PRIMARY KEY,
  name           TEXT UNIQUE NOT NULL,
  active         BOOLEAN DEFAULT TRUE
);

CREATE TABLE tag_evidence (
  entity_type    TEXT NOT NULL,    -- 'track' | 'artist' | 'release'
  entity_key     TEXT NOT NULL,    -- track_id / artist_id / album_id
  source         TEXT NOT NULL,    -- 'musicbrainz' | 'lastfm' | 'discogs' | 'theaudiodb'
  tag_raw        TEXT NOT NULL,    -- exact string from API
  tag_kind       TEXT NOT NULL,    -- 'genre' | 'mood'
  weight_raw     REAL,             -- source-provided weight, or 1.0
  observed_at    TIMESTAMP NOT NULL,
  PRIMARY KEY (entity_type, entity_key, source, tag_raw, tag_kind)
);

CREATE TABLE map_genre (
  source         TEXT NOT NULL,
  tag_raw        TEXT NOT NULL,
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  confidence     REAL NOT NULL,   -- 0–1
  PRIMARY KEY (source, tag_raw)
);


CREATE TABLE genre_hierarchy (
  parent_genre_id INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  child_genre_id  INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  PRIMARY KEY (parent_genre_id, child_genre_id),
  CHECK (parent_genre_id <> child_genre_id)
);
CREATE INDEX idx_genre_h_child  ON genre_hierarchy(child_genre_id);
CREATE INDEX idx_genre_h_parent ON genre_hierarchy(parent_genre_id);

CREATE TABLE map_mood (
  source         TEXT NOT NULL,
  tag_raw        TEXT NOT NULL,
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  confidence     REAL NOT NULL,
  PRIMARY KEY (source, tag_raw)
);

CREATE TABLE track_genres (
  track_id       TEXT NOT NULL REFERENCES dim_tracks(track_id),
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  score          REAL NOT NULL,         -- sums to 1.0 per track
  PRIMARY KEY (track_id, genre_id)
);

CREATE TABLE artist_genres (
  artist_id      TEXT NOT NULL REFERENCES dim_artists(artist_id),
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  PRIMARY KEY (artist_id, genre_id)
);

CREATE TABLE track_moods (
  track_id       TEXT NOT NULL REFERENCES dim_tracks(track_id),
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  score          REAL NOT NULL,         -- sums to 1.0 per track
  method_version TEXT NOT NULL,
  PRIMARY KEY (track_id, mood_id)
);

CREATE TABLE agg_user_genre_daily (
  user_id        TEXT NOT NULL REFERENCES dim_users(user_id),
  date           DATE NOT NULL,
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  plays          INTEGER NOT NULL,
  ms_listened    BIGINT NOT NULL,
  avg_score_weighted REAL NOT NULL,
  PRIMARY KEY (user_id, date, genre_id)
);

CREATE TABLE agg_user_genre_monthly (
  user_id        TEXT NOT NULL REFERENCES dim_users(user_id),
  month          TEXT NOT NULL,    -- 'YYYY-MM'
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  plays          INTEGER NOT NULL,
  ms_listened    BIGINT NOT NULL,
  avg_score_weighted REAL NOT NULL,
  PRIMARY KEY (user_id, month, genre_id)
);

CREATE TABLE agg_user_mood_daily (
  user_id        TEXT NOT NULL REFERENCES dim_users(user_id),
  date           DATE NOT NULL,
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  plays          INTEGER NOT NULL,
  ms_listened    BIGINT NOT NULL,
  avg_score_weighted REAL NOT NULL,
  PRIMARY KEY (user_id, date, mood_id)
);

CREATE TABLE agg_user_mood_monthly (
  user_id        TEXT NOT NULL REFERENCES dim_users(user_id),
  month          TEXT NOT NULL,    -- 'YYYY-MM'
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  plays          INTEGER NOT NULL,
  ms_listened    BIGINT NOT NULL,
  avg_score_weighted REAL NOT NULL,
  PRIMARY KEY (user_id, month, mood_id)
);

-- ========================
-- Indexes (query accelerators)
-- ========================
-- Note: DuckDB is columnar and often scans efficiently without indexes.
-- These indexes target frequent join/filter keys used throughout the app.

-- fact_plays: frequent filters by user_id and played_at, and joins by track_id
CREATE INDEX IF NOT EXISTS idx_fact_plays_user_time ON fact_plays(user_id, played_at);
CREATE INDEX IF NOT EXISTS idx_fact_plays_track     ON fact_plays(track_id);
-- Composite on the dedupe key helps anti-joins and lookups (matches UNIQUE columns)
CREATE INDEX IF NOT EXISTS idx_fact_plays_user_track_played ON fact_plays(user_id, track_id, played_at);

-- dim_tracks: join by album_id; lookups by track_id occur via fact_plays join
CREATE INDEX IF NOT EXISTS idx_dim_tracks_album ON dim_tracks(album_id);

-- bridge_track_artists: frequent join by (track_id, role='primary'); sometimes by (artist_id, role)
CREATE INDEX IF NOT EXISTS idx_bridge_track_role  ON bridge_track_artists(track_id, role);
CREATE INDEX IF NOT EXISTS idx_bridge_artist_role ON bridge_track_artists(artist_id, role);

-- artist_genres and track_genres: joins by both sides
CREATE INDEX IF NOT EXISTS idx_artist_genres_artist ON artist_genres(artist_id);
CREATE INDEX IF NOT EXISTS idx_artist_genres_genre  ON artist_genres(genre_id);
CREATE INDEX IF NOT EXISTS idx_track_genres_track   ON track_genres(track_id);
CREATE INDEX IF NOT EXISTS idx_track_genres_genre   ON track_genres(genre_id);

-- dim_albums: joins by album_id from dim_tracks
CREATE INDEX IF NOT EXISTS idx_dim_albums_album_id ON dim_albums(album_id);

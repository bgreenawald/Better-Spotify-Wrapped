-- ========================
-- Users & Auth (if needed)
-- ========================
CREATE TABLE dim_users (
  user_id         TEXT PRIMARY KEY,      -- internal UUID
  display_name    TEXT,
  country         TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================
-- Tracks, Albums, Artists
-- ========================
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

-- ========================
-- Facts: Listening history
-- ========================
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

-- ========================
-- Canonical vocabularies
-- ========================
CREATE TABLE dim_genres (
  genre_id       INTEGER PRIMARY KEY,
  name           TEXT UNIQUE NOT NULL,
  parent_genre_id INTEGER,
  level          INTEGER,
  active         BOOLEAN DEFAULT TRUE
);

CREATE TABLE dim_moods (
  mood_id        INTEGER PRIMARY KEY,
  name           TEXT UNIQUE NOT NULL,
  active         BOOLEAN DEFAULT TRUE
);

-- ========================
-- Raw evidence & mappings
-- ========================
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

CREATE TABLE map_mood (
  source         TEXT NOT NULL,
  tag_raw        TEXT NOT NULL,
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  confidence     REAL NOT NULL,
  PRIMARY KEY (source, tag_raw)
);

-- ========================
-- Normalized track-level
-- ========================
CREATE TABLE track_genres (
  track_id       TEXT NOT NULL REFERENCES dim_tracks(track_id),
  genre_id       INTEGER NOT NULL REFERENCES dim_genres(genre_id),
  score          REAL NOT NULL,         -- sums to 1.0 per track
  method_version TEXT NOT NULL,
  PRIMARY KEY (track_id, genre_id)
);

CREATE TABLE track_moods (
  track_id       TEXT NOT NULL REFERENCES dim_tracks(track_id),
  mood_id        INTEGER NOT NULL REFERENCES dim_moods(mood_id),
  score          REAL NOT NULL,         -- sums to 1.0 per track
  method_version TEXT NOT NULL,
  PRIMARY KEY (track_id, mood_id)
);

-- ========================
-- User rollups
-- ========================
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

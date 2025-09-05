import pandas as pd
import pytest

from src.metrics import metrics


@pytest.fixture
def sample_df():
    data = {
        "master_metadata_track_name": ["Song A", "Song B", "Song A", "Song C"],
        "master_metadata_album_artist_name": ["Artist 1", "Artist 2", "Artist 1", "Artist 3"],
        "spotify_track_uri": ["uri:1", "uri:2", "uri:1", "uri:3"],
        "artist_id": ["id1", "id2", "id1", "id3"],
        "artist_genres": [("pop",), ("rock",), ("pop",), ("jazz",)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_spotify_data():
    # Minimal mock for the SpotifyData object used in metrics.py
    class MockSpotifyData:
        def __init__(self):
            self.tracks = {
                "1": {
                    "album": {"id": "a1"},
                    "track_number": 1,
                    "artists": [{"id": "id1"}, {"id": "id2"}],
                },
                "2": {"album": {"id": "a2"}, "track_number": 1, "artists": [{"id": "id2"}]},
                "3": {"album": {"id": "a3"}, "track_number": 1, "artists": [{"id": "id3"}]},
            }
            self.albums = {
                "a1": {
                    "album_type": "album",
                    "total_tracks": 1,
                    "name": "Album 1",
                    "artists": [{"name": "Artist 1"}],
                    "release_date": "2020-01-01",
                },
                "a2": {
                    "album_type": "album",
                    "total_tracks": 1,
                    "name": "Album 2",
                    "artists": [{"name": "Artist 2"}],
                    "release_date": "2021-01-01",
                },
                "a3": {
                    "album_type": "album",
                    "total_tracks": 1,
                    "name": "Album 3",
                    "artists": [{"name": "Artist 3"}],
                    "release_date": "2022-01-01",
                },
            }
            self.artists = {
                "id1": {"name": "Artist 1", "genres": ["pop"]},
                "id2": {"name": "Artist 2", "genres": ["rock"]},
                "id3": {"name": "Artist 3", "genres": ["jazz"]},
            }

    return MockSpotifyData()


def test_get_most_played_tracks(sample_df):
    import duckdb

    con = duckdb.connect(":memory:")
    result = metrics.get_most_played_tracks(sample_df, con=con)
    assert not result.empty
    assert "track_name" in result.columns
    assert "artist" in result.columns
    assert "play_count" in result.columns
    assert result.iloc[0]["play_count"] >= result.iloc[-1]["play_count"]


def test_get_top_albums(sample_df):
    # Add matching track IDs to sample_df for compatibility
    sample_df = sample_df.copy()
    sample_df["spotify_track_uri"] = [
        "spotify:track:1",
        "spotify:track:2",
        "spotify:track:1",
        "spotify:track:3",
    ]

    # Set up an in-memory DuckDB with minimal schema and data
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE dim_tracks (
                track_id   TEXT PRIMARY KEY,
                album_id   TEXT
            );
            CREATE TABLE dim_albums (
                album_id     TEXT PRIMARY KEY,
                album_name   TEXT NOT NULL,
                release_year INT,
                label        TEXT
            );
            CREATE TABLE dim_artists (
                artist_id   TEXT PRIMARY KEY,
                artist_name TEXT NOT NULL
            );
            CREATE TABLE bridge_track_artists (
                track_id  TEXT NOT NULL,
                artist_id TEXT NOT NULL,
                role      TEXT NOT NULL
            );
            """
        )

        # New implementation expects v_plays_enriched; provide a minimal view
        # mapping played track_ids to album_ids for our synthetic dataset.
        con.execute(
            """
            CREATE VIEW v_plays_enriched AS
            SELECT t.track_id, t.album_id
            FROM dim_tracks t;
            """
        )

        # Insert three albums, each with >5 tracks so they are not filtered out
        con.executemany(
            "INSERT INTO dim_albums(album_id, album_name, release_year, label) VALUES (?, ?, ?, ?)",
            [
                ("a1", "Album 1", 2020, None),
                ("a2", "Album 2", 2021, None),
                ("a3", "Album 3", 2022, None),
            ],
        )
        con.executemany(
            "INSERT INTO dim_artists(artist_id, artist_name) VALUES (?, ?)",
            [("id1", "Artist 1"), ("id2", "Artist 2"), ("id3", "Artist 3")],
        )

        # Helper to create tracks and primary artist bridges
        def add_album_tracks(album_id: str, primary_artist: str, main_track: str) -> None:
            # 6 tracks per album including the main_track used by the sample_df
            track_ids = [main_track] + [f"{album_id}_t{i}" for i in range(2, 7)]
            con.executemany(
                "INSERT INTO dim_tracks(track_id, album_id) VALUES (?, ?)",
                [(tid, album_id) for tid in track_ids],
            )
            con.executemany(
                "INSERT INTO bridge_track_artists(track_id, artist_id, role) VALUES (?, ?, ?)",
                [(tid, primary_artist, "primary") for tid in track_ids],
            )

        add_album_tracks("a1", "id1", "1")
        add_album_tracks("a2", "id2", "2")
        add_album_tracks("a3", "id3", "3")

        # Run the album metric against the DuckDB backend
        result = metrics.get_top_albums(sample_df, con=con)
        assert isinstance(result, pd.DataFrame)
        assert {"album_name", "artist", "median_plays", "total_tracks", "tracks_played"}.issubset(
            result.columns
        )
        assert (result["median_plays"] >= 0).all()
        # Ensure we excluded short releases: all totals should be >= 6
        assert (result["total_tracks"] >= 6).all()
    finally:
        con.close()


def test_get_top_artist_genres(sample_df):
    # Add matching track IDs to sample_df for compatibility
    sample_df = sample_df.copy()
    sample_df["spotify_track_uri"] = [
        "spotify:track:1",
        "spotify:track:2",
        "spotify:track:1",
        "spotify:track:3",
    ]

    import duckdb

    con = duckdb.connect(":memory:")
    try:
        # Minimal schema needed for genre aggregation
        con.execute(
            """
            CREATE TABLE dim_artists (
                artist_id   TEXT PRIMARY KEY,
                artist_name TEXT NOT NULL
            );
            CREATE TABLE bridge_track_artists (
                track_id  TEXT NOT NULL,
                artist_id TEXT NOT NULL,
                role      TEXT NOT NULL
            );
            CREATE TABLE dim_genres (
                genre_id   INTEGER PRIMARY KEY,
                name       TEXT NOT NULL
            );
            CREATE TABLE artist_genres (
                artist_id  TEXT NOT NULL,
                genre_id   INTEGER NOT NULL
            );
            """
        )

        # Provide v_primary_artist_per_track view expected by the new SQL.
        # Deterministically pick the first primary artist by artist_name per track.
        con.execute(
            """
            CREATE VIEW v_primary_artist_per_track AS
            WITH ranked AS (
              SELECT b.track_id,
                     b.artist_id,
                     a.artist_name,
                     ROW_NUMBER() OVER (PARTITION BY b.track_id ORDER BY a.artist_name) AS rn
              FROM bridge_track_artists b
              JOIN dim_artists a ON a.artist_id = b.artist_id
              WHERE b.role = 'primary'
            )
            SELECT track_id, artist_id
            FROM ranked
            WHERE rn = 1;
            """
        )

        # Seed artists and bridges for tracks 1..3
        con.executemany(
            "INSERT INTO dim_artists(artist_id, artist_name) VALUES (?, ?)",
            [("id1", "Artist 1"), ("id2", "Artist 2"), ("id3", "Artist 3")],
        )
        con.executemany(
            "INSERT INTO bridge_track_artists(track_id, artist_id, role) VALUES (?, ?, ?)",
            [("1", "id1", "primary"), ("2", "id2", "primary"), ("3", "id3", "primary")],
        )
        # Seed genres and artist mappings
        con.executemany(
            "INSERT INTO dim_genres(genre_id, name) VALUES (?, ?)",
            [(1, "pop"), (2, "rock"), (3, "jazz")],
        )
        con.executemany(
            "INSERT INTO artist_genres(artist_id, genre_id) VALUES (?, ?)",
            [("id1", 1), ("id2", 2), ("id3", 3)],
        )

        result = metrics.get_top_artist_genres(sample_df, con=con)
        assert isinstance(result, pd.DataFrame)
        assert {"genre", "play_count"}.issubset(result.columns)
        assert result["play_count"].min() >= 0
    finally:
        con.close()


def test_get_listening_time_by_month():
    import duckdb

    # Build a tiny DataFrame across two months
    import pandas as pd

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-15 12:00:00",
                    "2024-02-02 09:00:00",
                ]
            ),
            "ms_played": [60_000, 120_000, 180_000],  # 1, 2, 3 minutes
            "master_metadata_track_name": ["A", "B", "C"],
            "master_metadata_album_artist_name": ["X", "Y", "Z"],
        }
    )

    con = duckdb.connect(":memory:")
    try:
        # Function moved to src.metrics.trends in the new implementation
        from src.metrics.trends import get_listening_time_by_month

        out = get_listening_time_by_month(df, con=con)
        assert list(out.columns) == [
            "month",
            "unique_tracks",
            "unique_artists",
            "total_hours",
            "avg_hours_per_day",
        ]
        # Two months expected
        assert set(out["month"]) == {"2024-01", "2024-02"}
        # Totals in hours > 0
        assert (out["total_hours"] > 0).all()
    finally:
        con.close()


def test_get_most_played_artists(sample_df):
    import duckdb

    con = duckdb.connect(":memory:")
    result = metrics.get_most_played_artists(sample_df, con=con)
    assert isinstance(result, pd.DataFrame)
    assert "artist" in result.columns
    assert "play_count" in result.columns
    assert "unique_tracks" in result.columns
    assert result["play_count"].min() >= 0

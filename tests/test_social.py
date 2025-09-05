import pandas as pd


def _setup_duckdb_with_minimal_schema(con):
    con.execute(
        """
        CREATE TABLE dim_tracks (
            track_id   TEXT PRIMARY KEY,
            track_name TEXT NOT NULL,
            album_id   TEXT
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
        CREATE TABLE fact_plays (
            user_id TEXT NOT NULL,
            track_id TEXT NOT NULL,
            played_at TIMESTAMP NOT NULL,
            skipped BOOLEAN,
            incognito_mode BOOLEAN,
            duration_ms INTEGER,
            reason_start TEXT,
            reason_end TEXT
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


def test_social_tracks_two_users_basic():
    import duckdb

    from src.metrics.social import compute_social_regions

    con = duckdb.connect(":memory:")
    try:
        _setup_duckdb_with_minimal_schema(con)
        # Tracks and artists
        con.executemany(
            "INSERT INTO dim_tracks(track_id, track_name, album_id) VALUES (?, ?, ?)",
            [("t1", "Track One", None), ("t2", "Track Two", None), ("t3", "Track Three", None)],
        )
        con.executemany(
            "INSERT INTO dim_artists(artist_id, artist_name) VALUES (?, ?)",
            [("a1", "Artist A"), ("a2", "Artist B")],
        )
        con.executemany(
            "INSERT INTO bridge_track_artists(track_id, artist_id, role) VALUES (?, ?, ?)",
            [("t1", "a1", "primary"), ("t2", "a1", "primary"), ("t3", "a2", "primary")],
        )

        # Plays: u1 plays t1 twice and t2 once; u2 plays t1 once and t3 once
        rows = []

        def add_play(user, tid, ts):
            rows.append((user, tid, ts, False, False, 60_000, "fw", "fw"))

        add_play("u1", "t1", "2024-01-01 10:00:00")
        add_play("u1", "t1", "2024-01-01 11:00:00")
        add_play("u1", "t2", "2024-01-02 10:00:00")
        add_play("u2", "t1", "2024-01-01 12:00:00")
        add_play("u2", "t3", "2024-01-03 09:00:00")
        con.executemany(
            "INSERT INTO fact_plays(user_id, track_id, played_at, skipped, incognito_mode, duration_ms, reason_start, reason_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

        out = compute_social_regions(
            con=con,
            users=["u1", "u2"],
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
            mode="tracks",
            exclude_december=True,
            remove_incognito=True,
        )
        assert out["users"] == ["u1", "u2"]
        regions = out["regions"]
        # Intersection region should include t1 only
        inter = regions.get("u1_u2", [])
        assert any(item["name"] == "Track One" for item in inter)
        assert all(item["name"] != "Track Two" for item in inter)
        # u1_only should include its top ranks with Track One first
        u1_only = regions.get("u1_only", [])
        assert len(u1_only) > 0
        assert u1_only[0]["name"] in {"Track One", "Track Two"}
    finally:
        con.close()


def test_social_artists_and_genres():
    import duckdb

    from src.metrics.social import compute_social_regions

    con = duckdb.connect(":memory:")
    try:
        _setup_duckdb_with_minimal_schema(con)
        con.executemany(
            "INSERT INTO dim_tracks(track_id, track_name, album_id) VALUES (?, ?, ?)",
            [("t1", "T1", None), ("t2", "T2", None)],
        )
        con.executemany(
            "INSERT INTO dim_artists(artist_id, artist_name) VALUES (?, ?)",
            [("a1", "Alpha"), ("a2", "Beta")],
        )
        con.executemany(
            "INSERT INTO bridge_track_artists(track_id, artist_id, role) VALUES (?, ?, ?)",
            [("t1", "a1", "primary"), ("t2", "a2", "primary")],
        )
        con.executemany(
            "INSERT INTO dim_genres(genre_id, name) VALUES (?, ?)", [(1, "Pop"), (2, "Rock")]
        )
        con.executemany(
            "INSERT INTO artist_genres(artist_id, genre_id) VALUES (?, ?)", [("a1", 1), ("a2", 2)]
        )
        rows = [
            ("u1", "t1", "2024-02-01 10:00:00", False, False, 60_000, "fw", "fw"),
            ("u2", "t2", "2024-02-01 10:00:00", False, False, 60_000, "fw", "fw"),
        ]
        con.executemany(
            "INSERT INTO fact_plays(user_id, track_id, played_at, skipped, incognito_mode, duration_ms, reason_start, reason_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

        artists = compute_social_regions(
            con=con, users=["u1", "u2"], start=None, end=None, mode="artists"
        )
        assert artists["regions"]["u1_u2"] == []  # no shared artists

        genres = compute_social_regions(
            con=con, users=["u1", "u2"], start=None, end=None, mode="genres"
        )
        # Different single-genre plays; no intersection expected
        assert genres["regions"]["u1_u2"] == []
    finally:
        con.close()

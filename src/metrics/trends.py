import pandas as pd

from src.api.api import SpotifyData


def get_listening_time_by_month(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total listening time by month.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'ms_played', 'master_metadata_track_name', and
            'master_metadata_album_artist_name' columns.

    Returns:
        pd.DataFrame: Monthly listening statistics with columns:
            - month (str): Year-month in 'YYYY-MM' format
            - unique_tracks (int): Number of unique tracks
            - unique_artists (int): Number of unique artists
            - total_hours (float): Total listening time in hours
            - avg_hours_per_day (float): Average listening time per day
    """
    # Group by month string and compute total playtime and unique counts
    monthly = (
        filtered_df.groupby(filtered_df["ts"].dt.strftime("%Y-%m"))
        .agg(
            {
                "ms_played": "sum",
                "master_metadata_track_name": "nunique",
                "master_metadata_album_artist_name": "nunique",
            }
        )
        .reset_index()
    )

    # Convert milliseconds to hours
    monthly["total_hours"] = monthly["ms_played"] / (1000 * 60 * 60)

    # Rename for clarity
    monthly.columns = [
        "month",
        "ms_played",
        "unique_tracks",
        "unique_artists",
        "total_hours",
    ]

    # Compute days in each month and average hours per day
    monthly["days_in_month"] = monthly["month"].apply(
        lambda m: pd.Period(m).days_in_month
    )
    monthly["avg_hours_per_day"] = monthly["total_hours"] / monthly["days_in_month"]

    # Round numeric columns
    monthly["total_hours"] = monthly["total_hours"].round(2)
    monthly["avg_hours_per_day"] = monthly["avg_hours_per_day"].round(2)

    # Drop intermediates, sort, and return
    result = monthly.drop(["ms_played", "days_in_month"], axis=1)
    result = result.sort_values("month").reset_index(drop=True)
    return result


def get_genre_trends(
    filtered_df: pd.DataFrame, spotify_data: SpotifyData
) -> pd.DataFrame:
    """Calculate genre listening trends over time.

    Optimized version using vectorized operations.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts' and 'spotify_track_uri' columns.
        spotify_data (SpotifyData): Spotify data with tracks and artists
            metadata.

    Returns:
        pd.DataFrame: Genre trends with columns:
            - month (str): 'YYYY-MM'
            - genre (str)
            - play_count (int)
            - percentage (float)
            - top_artists (str)
            - rank (float): Rank of genre by plays within month
    """
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")

    # Build mapping of track URI to artist names and genres
    track_mappings = []
    for uri in df["spotify_track_uri"].unique():
        if not uri:
            continue
        track_id = uri.split(":")[-1]
        track_meta = spotify_data.tracks.get(track_id)
        if not track_meta:
            continue

        artist_id = track_meta["artists"][0]["id"]
        artist_meta = spotify_data.artists.get(artist_id)
        if not artist_meta:
            continue

        genres = artist_meta.get("genres", [])
        for genre in genres:
            track_mappings.append(
                {
                    "spotify_track_uri": uri,
                    "artist_name": artist_meta["name"],
                    "genre": genre,
                }
            )

    # Return empty frame if no mappings
    if not track_mappings:
        return pd.DataFrame(
            columns=[
                "month",
                "genre",
                "play_count",
                "percentage",
                "top_artists",
                "rank",
            ]
        )

    track_info = pd.DataFrame(track_mappings)

    # Merge plays with genre info
    merged = df.merge(track_info, on="spotify_track_uri")

    # Count plays per artist-genre-month
    genre_artist_plays = (
        merged.groupby(["month", "genre", "artist_name"])
        .size()
        .reset_index(name="artist_plays")
    )

    # Sum to get play_count per genre-month
    genre_totals = (
        genre_artist_plays.groupby(["month", "genre"])["artist_plays"]
        .sum()
        .reset_index(name="play_count")
    )

    # Total plays per month for percentage calc
    monthly_totals = (
        genre_totals.groupby("month")["play_count"]
        .sum()
        .reset_index(name="total_plays")
    )

    # Determine top 2 artists by plays per genre-month
    top_artists = (
        genre_artist_plays.sort_values("artist_plays", ascending=False)
        .groupby(["month", "genre"])
        .agg(
            {
                "artist_name": lambda names: ", ".join(names.head(2)),
                "artist_plays": lambda counts: ", ".join(
                    f"({cnt} plays)" for cnt in counts.head(2)
                ),
            }
        )
        .reset_index()
    )

    # Combine artist names with play counts
    top_artists["top_artists"] = top_artists.apply(
        lambda row: " ".join(
            f"{name} {plays}"
            for name, plays in zip(
                row["artist_name"].split(", "), row["artist_plays"].split(", ")
            )
        ),
        axis=1,
    )

    # Merge totals, percentages, and top artists
    result = genre_totals.merge(monthly_totals, on="month").merge(
        top_artists[["month", "genre", "top_artists"]], on=["month", "genre"]
    )

    result["percentage"] = (result["play_count"] / result["total_plays"] * 100).round(2)

    # Clean up and sort
    result = result.drop("total_plays", axis=1)
    result = result.sort_values(["month", "play_count"], ascending=[True, False])

    # Add rank within each month
    result["rank"] = result.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return result.reset_index(drop=True)


def get_artist_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate artist listening trends over time.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_album_artist_name',
            'master_metadata_track_name', and 'ms_played' columns.

    Returns:
        pd.DataFrame: Artist trends with columns:
            - month (str)
            - artist (str)
            - play_count (int)
            - unique_tracks (int)
            - percentage (float)
            - avg_duration_min (float)
            - top_tracks (str)
            - rank (float)
    """
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")

    # Count plays per track per artist-month
    track_counts = (
        df.groupby(
            ["month", "master_metadata_album_artist_name", "master_metadata_track_name"]
        )
        .size()
        .reset_index(name="track_plays")
    )

    # Rank and select top 2 tracks per artist-month
    track_counts["track_rank"] = track_counts.groupby(
        ["month", "master_metadata_album_artist_name"]
    )["track_plays"].rank(method="dense", ascending=False)
    top_tracks_df = track_counts[track_counts["track_rank"] <= 2]

    # Aggregate top tracks strings
    top_tracks_by_artist = (
        top_tracks_df.sort_values("track_plays", ascending=False)
        .groupby(["month", "master_metadata_album_artist_name"])
        .apply(
            lambda grp: ", ".join(
                f"{row['master_metadata_track_name']} ({row['track_plays']} plays)"
                for _, row in grp.head(2).iterrows()
            )
        )
        .reset_index(name="top_tracks")
    )

    # Compute artist metrics: play count, unique tracks, avg duration
    metrics = df.groupby(["month", "master_metadata_album_artist_name"]).agg(
        {
            "master_metadata_track_name": ["count", "nunique"],
            "ms_played": "mean",
        }
    )
    metrics.columns = ["play_count", "unique_tracks", "avg_duration_ms"]
    metrics = metrics.reset_index()

    # Compute monthly totals for percentage
    monthly_totals = (
        metrics.groupby("month")["play_count"].sum().reset_index(name="total_plays")
    )
    metrics = metrics.merge(monthly_totals, on="month")
    metrics["percentage"] = (
        metrics["play_count"] / metrics["total_plays"] * 100
    ).round(2)

    # Convert duration to minutes
    metrics["avg_duration_min"] = (metrics["avg_duration_ms"] / (1000 * 60)).round(2)

    # Combine with top tracks
    result = metrics.merge(
        top_tracks_by_artist,
        on=["month", "master_metadata_album_artist_name"],
        how="left",
    )

    # Clean up and rename
    result = result.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(columns={"master_metadata_album_artist_name": "artist"})

    # Sort and rank
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return result.reset_index(drop=True)


def get_track_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track listening trends over time.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_track_name',
            'master_metadata_album_artist_name', and 'ms_played' columns.

    Returns:
        pd.DataFrame: Track trends with columns:
            - month (str)
            - track (str)
            - artist (str)
            - track_artist (str)
            - play_count (int)
            - percentage (float)
            - avg_duration_min (float)
            - rank (float)
    """
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")

    # Combine track name and artist
    df["track_artist"] = (
        df["master_metadata_track_name"]
        + " - "
        + df["master_metadata_album_artist_name"]
    )

    # Compute play count and avg duration per track-artist-month
    track_metrics = df.groupby(
        [
            "month",
            "track_artist",
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
        ]
    ).agg(
        {
            "track_artist": "count",
            "ms_played": "mean",
        }
    )
    track_metrics.columns = ["play_count", "avg_duration_ms"]
    track_metrics = track_metrics.reset_index()

    # Monthly totals for percentage
    monthly_totals = (
        track_metrics.groupby("month")["play_count"]
        .sum()
        .reset_index(name="total_plays")
    )
    track_metrics = track_metrics.merge(monthly_totals, on="month")
    track_metrics["percentage"] = (
        track_metrics["play_count"] / track_metrics["total_plays"] * 100
    ).round(2)

    # Convert duration to minutes
    track_metrics["avg_duration_min"] = (
        track_metrics["avg_duration_ms"] / (1000 * 60)
    ).round(2)

    # Clean up and rename columns
    result = track_metrics.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        }
    )

    # Sort and rank
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return result.reset_index(drop=True)

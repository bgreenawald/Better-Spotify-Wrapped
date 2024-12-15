import pandas as pd

from src.api.api import SpotifyData


def get_listening_time_by_month(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total listening time by month from a filtered Spotify listening history DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()

    Returns:
        pd.DataFrame: DataFrame containing monthly listening stats with columns:
                     'month', 'total_hours', 'unique_tracks', 'unique_artists'
    """
    # Group by month and calculate metrics
    monthly_stats = (
        filtered_df.groupby(filtered_df["ts"].dt.strftime("%Y-%m"))
        .agg(
            {
                "ms_played": "sum",  # Total listening time
                "master_metadata_track_name": "nunique",  # Unique tracks
                "master_metadata_album_artist_name": "nunique",  # Unique artists
            }
        )
        .reset_index()
    )

    # Convert milliseconds to hours
    monthly_stats["total_hours"] = monthly_stats["ms_played"] / (1000 * 60 * 60)

    # Rename columns for clarity
    monthly_stats.columns = [
        "month",
        "ms_played",
        "unique_tracks",
        "unique_artists",
        "total_hours",
    ]

    # Calculate average daily listening time for each month
    monthly_stats["days_in_month"] = monthly_stats["month"].apply(
        lambda x: pd.Period(x).days_in_month
    )
    monthly_stats["avg_hours_per_day"] = (
        monthly_stats["total_hours"] / monthly_stats["days_in_month"]
    )

    # Round numeric columns
    monthly_stats["total_hours"] = monthly_stats["total_hours"].round(2)
    monthly_stats["avg_hours_per_day"] = monthly_stats["avg_hours_per_day"].round(2)

    # Drop intermediate columns and sort by month
    final_stats = monthly_stats.drop(["ms_played", "days_in_month"], axis=1)
    final_stats = final_stats.sort_values("month")

    return final_stats


def get_genre_trends(
    filtered_df: pd.DataFrame, spotify_data: SpotifyData
) -> pd.DataFrame:
    """
    Calculate genre listening trends over time from a filtered Spotify listening history DataFrame.
    Optimized version using vectorized operations.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()
        spotify_data (SpotifyData): Container with Spotify API data including tracks, artists, and albums

    Returns:
        pd.DataFrame: DataFrame containing genre trends with columns:
                     'month', 'genre', 'play_count', 'percentage', 'top_artists'
    """
    # Convert timestamp to month format
    filtered_df = filtered_df.copy()
    filtered_df["month"] = filtered_df["ts"].dt.strftime("%Y-%m")

    # Create a DataFrame mapping tracks to their artists and genres
    track_mappings = []
    for track_uri in filtered_df["spotify_track_uri"].unique():
        track_id = track_uri.split(":")[-1]
        if track_id in spotify_data.tracks:
            artist_id = spotify_data.tracks[track_id]["artists"][0]["id"]
            if artist_id in spotify_data.artists:
                artist_data = spotify_data.artists[artist_id]
                genres = artist_data.get("genres", [])
                if genres:
                    track_mappings.extend(
                        [
                            {
                                "spotify_track_uri": track_uri,  # Keep the full URI
                                "artist_name": artist_data["name"],
                                "genre": genre,
                            }
                            for genre in genres
                        ]
                    )

    if not track_mappings:
        return pd.DataFrame(
            columns=["month", "genre", "play_count", "percentage", "top_artists"]
        )

    # Convert to DataFrame
    track_info_df = pd.DataFrame(track_mappings)

    # Merge with listening history to get all plays
    merged_df = filtered_df.merge(track_info_df, on="spotify_track_uri")

    # Calculate genre play counts and artist contributions per month
    genre_plays = (
        merged_df.groupby(["month", "genre", "artist_name"])
        .size()
        .reset_index(name="artist_plays")
    )

    # Get total plays per genre per month
    genre_totals = (
        genre_plays.groupby(["month", "genre"])["artist_plays"]
        .sum()
        .reset_index(name="play_count")
    )

    # Calculate monthly totals for percentage calculation
    monthly_totals = (
        genre_totals.groupby("month")["play_count"]
        .sum()
        .reset_index(name="total_plays")
    )

    # Get top artists per genre per month
    top_artists = (
        genre_plays.sort_values("artist_plays", ascending=False)
        .groupby(["month", "genre"])
        .agg(
            {
                "artist_name": lambda x: ", ".join(f"{artist}" for artist in x.head(2)),
                "artist_plays": lambda x: ", ".join(
                    f"({plays} plays)" for plays in x.head(2)
                ),
            }
        )
        .reset_index()
    )

    # Combine artist names and play counts
    top_artists["top_artists"] = top_artists.apply(
        lambda row: " ".join(
            [
                n + " " + p
                for n, p in zip(
                    row["artist_name"].split(", "), row["artist_plays"].split(", ")
                )
            ]
        ),
        axis=1,
    )

    # Merge all data together
    results_df = genre_totals.merge(monthly_totals, on="month").merge(
        top_artists[["month", "genre", "top_artists"]], on=["month", "genre"]
    )

    # Calculate percentages
    results_df["percentage"] = (
        results_df["play_count"] / results_df["total_plays"] * 100
    ).round(2)

    # Clean up and sort
    results_df = results_df.drop("total_plays", axis=1)
    results_df = results_df.sort_values(
        ["month", "play_count"], ascending=[True, False]
    )

    # Add rank within each month
    results_df["rank"] = results_df.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return results_df


def get_artist_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate artist listening trends over time from a filtered Spotify listening history DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()

    Returns:
        pd.DataFrame: DataFrame containing artist trends with columns:
                     'month', 'artist', 'play_count', 'percentage', 'unique_tracks',
                     'avg_duration_min', 'top_tracks'
    """
    # Convert timestamp to month format
    filtered_df["month"] = filtered_df["ts"].dt.strftime("%Y-%m")

    # Calculate top tracks for each artist and month in one go
    top_tracks_df = (
        filtered_df.groupby(
            ["month", "master_metadata_album_artist_name", "master_metadata_track_name"]
        )
        .size()
        .reset_index(name="track_plays")
    )

    # Sort and get top 3 tracks per artist per month
    top_tracks_df["track_rank"] = top_tracks_df.groupby(
        ["month", "master_metadata_album_artist_name"]
    )["track_plays"].rank(method="dense", ascending=False)

    top_tracks_df = top_tracks_df[top_tracks_df["track_rank"] <= 2]

    # Create the top tracks string for each artist and month
    top_tracks_agg = (
        top_tracks_df.sort_values("track_plays", ascending=False)
        .groupby(["month", "master_metadata_album_artist_name"])
        .apply(
            lambda x: ", ".join(
                f"{row['master_metadata_track_name']} ({row['track_plays']} plays)"
                for _, row in x.iloc[:2].iterrows()
            )
        )
        .reset_index(name="top_tracks")
    )

    # Calculate main artist metrics
    artist_metrics = filtered_df.groupby(
        ["month", "master_metadata_album_artist_name"]
    ).agg(
        {
            "master_metadata_track_name": [
                "count",
                "nunique",
            ],  # play count and unique tracks
            "ms_played": "mean",  # average duration
        }
    )

    # Flatten column names
    artist_metrics.columns = ["play_count", "unique_tracks", "avg_duration_ms"]
    artist_metrics = artist_metrics.reset_index()

    # Calculate total plays per month for percentage calculation
    monthly_totals = (
        artist_metrics.groupby("month")["play_count"]
        .sum()
        .reset_index(name="total_plays")
    )

    # Merge monthly totals back to calculate percentages
    artist_metrics = artist_metrics.merge(monthly_totals, on="month")
    artist_metrics["percentage"] = (
        artist_metrics["play_count"] / artist_metrics["total_plays"] * 100
    ).round(2)

    # Convert average duration to minutes
    artist_metrics["avg_duration_min"] = (
        artist_metrics["avg_duration_ms"] / (1000 * 60)
    ).round(2)

    # Merge with top tracks data
    final_df = artist_metrics.merge(
        top_tracks_agg, on=["month", "master_metadata_album_artist_name"], how="left"
    )

    # Clean up and rename columns
    final_df = final_df.drop(["avg_duration_ms", "total_plays"], axis=1)
    final_df = final_df.rename(columns={"master_metadata_album_artist_name": "artist"})

    # Sort by month and play count
    final_df = final_df.sort_values(["month", "play_count"], ascending=[True, False])

    # Add rank within each month
    final_df["rank"] = final_df.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return final_df


def get_track_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate track listening trends over time from a filtered Spotify listening history DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()

    Returns:
        pd.DataFrame: DataFrame containing track trends with columns:
                     'month', 'track', 'artist', 'play_count', 'percentage',
                     'avg_duration_min'
    """
    # Convert timestamp to month format
    filtered_df["month"] = filtered_df["ts"].dt.strftime("%Y-%m")

    # Calculate main track metrics
    track_metrics = filtered_df.groupby(
        ["month", "master_metadata_track_name", "master_metadata_album_artist_name"]
    ).agg(
        {
            "master_metadata_track_name": "count",  # play count
            "ms_played": "mean",  # average duration
        }
    )

    # Flatten column names
    track_metrics.columns = ["play_count", "avg_duration_ms"]
    track_metrics = track_metrics.reset_index()

    # Calculate total plays per month for percentage calculation
    monthly_totals = (
        track_metrics.groupby("month")["play_count"]
        .sum()
        .reset_index(name="total_plays")
    )

    # Merge monthly totals back to calculate percentages
    track_metrics = track_metrics.merge(monthly_totals, on="month")
    track_metrics["percentage"] = (
        track_metrics["play_count"] / track_metrics["total_plays"] * 100
    ).round(2)

    # Convert average duration to minutes
    track_metrics["avg_duration_min"] = (
        track_metrics["avg_duration_ms"] / (1000 * 60)
    ).round(2)

    # Clean up and rename columns
    final_df = track_metrics.drop(["avg_duration_ms", "total_plays"], axis=1)
    final_df = final_df.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        }
    )

    # Sort by month and play count
    final_df = final_df.sort_values(["month", "play_count"], ascending=[True, False])

    # Add rank within each month
    final_df["rank"] = final_df.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    return final_df

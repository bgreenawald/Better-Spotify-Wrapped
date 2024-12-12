from typing import Dict, List

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

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()
        spotify_data (SpotifyData): Container with Spotify API data including tracks, artists, and albums

    Returns:
        pd.DataFrame: DataFrame containing genre trends with columns:
                     'month', 'genre', 'play_count', 'percentage'
    """
    # Convert timestamp to month format
    filtered_df["month"] = filtered_df["ts"].dt.strftime("%Y-%m")

    # Create mapping of tracks to genres
    track_genres: Dict[str, List[str]] = {}

    for _, row in filtered_df.iterrows():
        track_uri = row["spotify_track_uri"]
        track_id = track_uri.split(":")[-1]

        # Skip if track not in Spotify data
        if track_id not in spotify_data.tracks:
            continue

        # Get artist ID for this track
        artist_id = spotify_data.tracks[track_id]["artists"][0]["id"]

        # Skip if artist not in Spotify data
        if artist_id not in spotify_data.artists:
            continue

        # Get artist genres
        artist_genres = spotify_data.artists[artist_id].get("genres", [])

        if artist_genres:
            track_genres[track_uri] = artist_genres

    # Create a list to store all genre occurrences
    genre_data = []

    # Process each month's data
    for month in filtered_df["month"].unique():
        month_df = filtered_df[filtered_df["month"] == month]

        # Count total tracks played this month
        total_tracks = 0
        month_genre_counts = {}

        # Count genre occurrences for this month
        for _, row in month_df.iterrows():
            track_uri = row["spotify_track_uri"]
            if track_uri in track_genres:
                total_tracks += 1
                for genre in track_genres[track_uri]:
                    month_genre_counts[genre] = month_genre_counts.get(genre, 0) + 1

        # Calculate percentages and add to results
        if total_tracks > 0:
            for genre, count in month_genre_counts.items():
                genre_data.append(
                    {
                        "month": month,
                        "genre": genre,
                        "play_count": count,
                        "percentage": round(count / total_tracks * 100, 2),
                    }
                )

    # Convert to DataFrame
    trends_df = pd.DataFrame(genre_data)

    # Sort by month and play count
    trends_df = trends_df.sort_values(["month", "play_count"], ascending=[True, False])

    # Add rank within each month
    trends_df["rank"] = trends_df.groupby("month")["play_count"].rank(
        method="dense", ascending=False
    )

    # Calculate month-over-month change
    trends_df["prev_percentage"] = trends_df.groupby("genre")["percentage"].shift(1)
    trends_df["percentage_change"] = (
        trends_df["percentage"] - trends_df["prev_percentage"]
    ).round(2)

    return trends_df

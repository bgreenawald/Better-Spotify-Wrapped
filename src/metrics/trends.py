import pandas as pd


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

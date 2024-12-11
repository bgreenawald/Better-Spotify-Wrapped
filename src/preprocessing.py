from typing import Optional

import pandas as pd


def filter_songs(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    exclude_december: bool = True,
    remove_incognito: bool = True,
) -> pd.DataFrame:
    """
    Filter the Spotify listening history DataFrame based on various criteria.

    Args:
        df (pd.DataFrame): DataFrame containing Spotify listening history
        start_date (pd.Timestamp): Start date (inclusive) for filtering
        end_date (pd.Timestamp): End date (inclusive) for filtering
        exclude_december (bool): If True, only include songs from Jan 1 to Dec 1
        remove_incognito (bool): If True, remove songs played in incognito mode

    Returns:
        pd.DataFrame: Filtered DataFrame containing only songs matching criteria
    """
    # Start with basic song filter (excluding podcasts)
    filtered_df = df[df["episode_name"].isna()]

    # Filter by date range
    if start_date:
        filtered_df = filtered_df[(filtered_df["ts"] >= start_date)]

    if end_date:
        filtered_df = filtered_df[(filtered_df["ts"] <= end_date)]

    # Optionally exclude December
    if exclude_december:
        filtered_df = filtered_df[filtered_df["ts"].dt.month < 12]

    # Remove skipped songs
    filtered_df = filtered_df[~filtered_df["skipped"]]

    # Remove tracks with no playtime
    filtered_df = filtered_df[filtered_df["ms_played"] > 0]

    # Optionally remove incognito mode songs
    if remove_incognito:
        filtered_df = filtered_df[~filtered_df["incognito_mode"]]

    return filtered_df

import pandas as pd


def filter_songs(
    df: pd.DataFrame,
    year: int,
    exclude_december: bool = True,
    remove_incognito: bool = True,
) -> pd.DataFrame:
    """
    Filter the Spotify listening history DataFrame based on various criteria.

    Args:
        df (pd.DataFrame): DataFrame containing Spotify listening history
        year (int): Year to filter for
        exclude_december (bool): If True, only include songs from Jan 1 to Dec 1
        remove_incognito (bool): If True, remove songs played in incognito mode

    Returns:
        pd.DataFrame: Filtered DataFrame containing only songs matching criteria
    """
    # Start with basic song filter (excluding podcasts)
    filtered_df = df[df["episode_name"].isna()]

    # Filter by year
    filtered_df = filtered_df[filtered_df["ts"].dt.year == year]

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

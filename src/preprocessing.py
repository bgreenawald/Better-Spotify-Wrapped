from typing import Optional

import pandas as pd

from src.api.api import SpotifyData


def add_api_data(df: pd.DataFrame, api_data: SpotifyData) -> pd.DataFrame:
    """
    Add API data to the listening history DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing listening history
        api_data (SpotifyData): Container with API data

    Returns:
        pd.DataFrame: DataFrame with API data added
    """
    # Add the artist and album ids
    df["track_id"] = df["spotify_track_uri"].apply(
        lambda uri: uri.split(":")[-1] if uri else None
    )
    df["album_id"] = df["track_id"].apply(
        lambda track_id: api_data.tracks[track_id]["album"]["id"]
        if track_id and track_id in api_data.tracks
        else None
    )
    df["artist_id"] = df["track_id"].apply(
        lambda track_id: api_data.tracks[track_id]["artists"][0]["id"]
        if track_id and track_id in api_data.tracks
        else None
    )

    # Add the artist genres
    df["artist_genres"] = df["artist_id"].apply(
        lambda artist_id: tuple(api_data.artists[artist_id]["genres"])
        if artist_id
        else None
    )

    return df


def filter_songs(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    exclude_december: bool = True,
    remove_incognito: bool = True,
    excluded_tracks: Optional[list[str]] = None,
    excluded_artists: Optional[list[str]] = None,
    excluded_albums: Optional[list[str]] = None,
    exluded_genres: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Filter the Spotify listening history DataFrame based on various criteria.

    Args:
        df (pd.DataFrame): DataFrame containing Spotify listening history
        start_date (pd.Timestamp): Start date (inclusive) for filtering
        end_date (pd.Timestamp): End date (inclusive) for filtering
        exclude_december (bool): If True, only include songs from Jan 1 to Dec 1
        remove_incognito (bool): If True, remove songs played in incognito mode
        excluded_tracks (list[str]): List of track names to exclude
        excluded_artists (list[str]): List of artist names to exclude
        excluded_albums (list[str]): List of album names to exclude
        exluded_genres (list[str]): List of genres to exclude

    Returns:
        pd.DataFrame: Filtered DataFrame containing only songs matching criteria
    """
    # Start with basic song filter (excluding podcasts)
    filtered_df = df[df["episode_name"].isna()]

    # Exclude tracks without track ids
    filtered_df = filtered_df[filtered_df["spotify_track_uri"].notna()]

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

    # Filter out reason start and end unknown
    filtered_df = filtered_df[
        (filtered_df["reason_start"] != "unknown")
        & (filtered_df["reason_end"] != "unknown")
    ]

    # Optionally remove incognito mode songs
    if remove_incognito:
        filtered_df = filtered_df[~filtered_df["incognito_mode"]]

    # Optionally exclude tracks, artists, and albums
    if excluded_tracks:
        filtered_df = filtered_df[
            ~filtered_df["master_metadata_track_name"].isin(excluded_tracks)
        ]

    if excluded_artists:
        filtered_df = filtered_df[
            ~filtered_df["master_metadata_album_artist_name"].isin(excluded_artists)
        ]

    if excluded_albums:
        filtered_df = filtered_df[
            ~filtered_df["master_metadata_album_album_name"].isin(excluded_albums)
        ]

    if exluded_genres:
        filtered_df = filtered_df[
            ~filtered_df["artist_genres"].apply(
                lambda genres: any(genre in genres for genre in exluded_genres)
                if genres
                else False
            )
        ]

    return filtered_df

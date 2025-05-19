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
    # Vectorized extraction of track_id
    df["track_id"] = df["spotify_track_uri"].str.split(":").str[-1]

    # Build track_id → album_id and artist_id mapping dicts
    track_to_album = {tid: tdata["album"]["id"] for tid, tdata in api_data.tracks.items()}
    track_to_artist = {tid: tdata["artists"][0]["id"] for tid, tdata in api_data.tracks.items()}

    df["album_id"] = df["track_id"].map(track_to_album)
    df["artist_id"] = df["track_id"].map(track_to_artist)

    # Build artist_id → genres mapping dict
    artist_to_genres = {aid: tuple(ad["genres"]) for aid, ad in api_data.artists.items()}
    df["artist_genres"] = df["artist_id"].map(artist_to_genres)

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

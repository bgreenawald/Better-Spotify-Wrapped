from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.api.api import SpotifyData


def get_most_played_tracks(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the most played tracks from a filtered Spotify listening history.

    Groups tracks by track name and artist, counts plays, and computes play
    percentages.

    Args:
        filtered_df (pd.DataFrame): DataFrame already filtered by filter_songs().

    Returns:
        pd.DataFrame: Tracks sorted by play count with columns:
            'track_artist', 'track_name', 'artist', 'artist_id',
            'artist_genres', 'play_count', 'percentage'.
    """
    df = filtered_df.copy()
    # Combine track name and artist for grouping
    df["track_artist"] = (
        df["master_metadata_track_name"]
        + " - "
        + df["master_metadata_album_artist_name"]
    )

    # Count plays per track
    grouped = (
        df.groupby(
            [
                "track_artist",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Sort by descending play count
    sorted_df = grouped.sort_values("play_count", ascending=False)

    # Calculate percentage of total plays
    total_plays = sorted_df["play_count"].sum()
    sorted_df["percentage"] = sorted_df["play_count"] / total_plays

    # Rename columns for clarity
    sorted_df.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )

    return sorted_df


def get_top_albums(
    filtered_df: pd.DataFrame, spotify_data: SpotifyData
) -> pd.DataFrame:
    """Calculate top albums based on median song plays, including songs with zero plays.

    Considers only full albums present in the filtered listening history.

    Args:
        filtered_df (pd.DataFrame): DataFrame already filtered by filter_songs().
        spotify_data (SpotifyData): Spotify API data containing tracks and albums.

    Returns:
        pd.DataFrame: Albums sorted by median play count with columns:
            'album_name', 'artist', 'median_plays', 'total_tracks',
            'tracks_played', 'release_date'.
    """
    # Identify albums in listening history
    album_ids_in_history: Set[str] = set()
    for uri in filtered_df["spotify_track_uri"].unique():
        track_id = uri.split(":")[-1]
        track_meta = spotify_data.tracks.get(track_id)
        if not track_meta:
            continue
        album_id = track_meta["album"]["id"]
        album_meta = spotify_data.albums.get(album_id)
        if album_meta and album_meta.get("album_type") == "album":
            album_ids_in_history.add(album_id)

    # Map each track URI to its album and track index
    track_to_album: Dict[str, Tuple[str, int]] = {}
    for track_id, track_meta in spotify_data.tracks.items():
        album_id = track_meta["album"]["id"]
        if album_id not in album_ids_in_history:
            continue
        uri = f"spotify:track:{track_id}"
        # Convert track number to zero-based index
        track_to_album[uri] = (album_id, track_meta["track_number"] - 1)

    # Count plays per track URI
    play_counts = filtered_df.groupby("spotify_track_uri").size().to_dict()

    # Initialize play counts for all tracks in each album
    album_play_lists: Dict[str, List[int]] = {
        album_id: [0] * spotify_data.albums[album_id]["total_tracks"]
        for album_id in album_ids_in_history
    }

    # Populate actual play counts
    for uri, count in play_counts.items():
        if uri in track_to_album:
            album_id, idx = track_to_album[uri]
            album_play_lists[album_id][idx] = count

    # Compile album statistics
    album_stats: List[Dict[str, object]] = []
    for album_id, plays in album_play_lists.items():
        album_meta = spotify_data.albums[album_id]
        total_tracks = album_meta["total_tracks"]
        album_stats.append(
            {
                "album_name": album_meta["name"],
                "artist": album_meta["artists"][0]["name"],
                "median_plays": float(np.median(plays)),
                "total_tracks": total_tracks,
                "tracks_played": sum(1 for p in plays if p > 0),
                "release_date": album_meta["release_date"],
            }
        )

    # Return empty DataFrame if no albums found
    if not album_stats:
        return pd.DataFrame()

    # Create DataFrame and sort by median plays
    result = pd.DataFrame(album_stats)
    return result.sort_values("median_plays", ascending=False)


def get_top_artist_genres(
    filtered_df: pd.DataFrame,
    spotify_data: SpotifyData,
    unique_tracks: bool = False,
    top_artists_per_genre: int = 2,
) -> pd.DataFrame:
    """Calculate the most common artist genres in the listening history.

    Counts genre occurrences per track (optionally unique) and lists top
    artists by play count for each genre.

    Args:
        filtered_df (pd.DataFrame): DataFrame already filtered by filter_songs().
        spotify_data (SpotifyData): Spotify API data containing tracks and artists.
        unique_tracks (bool): If True, consider each track URI only once.
        top_artists_per_genre (int): Number of top artists to include per genre.

    Returns:
        pd.DataFrame: Genres sorted by frequency with columns:
            'genre', 'play_count', 'percentage', 'top_artists'.
    """
    # Select track URIs, deduplicating if requested
    track_uris = (
        filtered_df["spotify_track_uri"].unique()
        if unique_tracks
        else filtered_df["spotify_track_uri"]
    )

    genre_counts: Dict[str, int] = {}
    genre_artists: Dict[str, Dict[str, int]] = {}
    total_tracked = 0

    for uri in track_uris:
        track_id = uri.split(":")[-1]
        track_meta = spotify_data.tracks.get(track_id)
        if not track_meta:
            continue

        artist_id = track_meta["artists"][0]["id"]
        artist_meta = spotify_data.artists.get(artist_id)
        if not artist_meta:
            continue

        genres = artist_meta.get("genres", [])
        if not genres:
            continue

        total_tracked += 1
        artist_name = artist_meta["name"]

        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            artists_for_genre = genre_artists.setdefault(genre, {})
            artists_for_genre[artist_name] = artists_for_genre.get(artist_name, 0) + 1

    # Build rows for DataFrame
    rows: List[Dict[str, object]] = []
    for genre, count in genre_counts.items():
        top_artists = sorted(genre_artists[genre].items(), key=lambda x: (-x[1], x[0]))[
            :top_artists_per_genre
        ]
        formatted_artists = ", ".join(
            f"{name} ({plays} plays)" for name, plays in top_artists
        )
        rows.append(
            {
                "genre": genre,
                "play_count": count,
                "top_artists": formatted_artists,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["genre", "play_count", "percentage", "top_artists"]
        )

    genre_df = pd.DataFrame(rows)
    genre_df["percentage"] = (genre_df["play_count"] / total_tracked * 100).round(2)

    return genre_df.sort_values("play_count", ascending=False)


def get_most_played_artists(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the most played artists from a filtered Spotify listening history.

    Counts total plays and unique tracks per artist, with play percentages.

    Args:
        filtered_df (pd.DataFrame): DataFrame filtered by filter_songs().

    Returns:
        pd.DataFrame: Artists sorted by play count with columns:
            'artist', 'artist_id', 'artist_genres', 'play_count',
            'unique_tracks', 'percentage'.
    """
    df = filtered_df.copy()

    # Total plays per artist
    plays = (
        df.groupby(
            [
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Unique tracks per artist
    uniques = (
        df.groupby(
            [
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )["master_metadata_track_name"]
        .nunique()
        .reset_index(name="unique_tracks")
    )

    # Merge play counts and unique track counts
    stats = pd.merge(
        plays,
        uniques,
        on=[
            "master_metadata_album_artist_name",
            "artist_id",
            "artist_genres",
        ],
    )

    # Sort and compute percentage
    stats = stats.sort_values("play_count", ascending=False)
    total_plays = stats["play_count"].sum()
    stats["percentage"] = (stats["play_count"] / total_plays * 100).round(2)

    # Rename columns for clarity
    stats.rename(
        columns={"master_metadata_album_artist_name": "artist"},
        inplace=True,
    )

    return stats


def get_playcount_by_day(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily play counts from a filtered Spotify listening history.

    Groups by date, track, and artist to count plays per day.

    Args:
        filtered_df (pd.DataFrame): DataFrame filtered by filter_songs().

    Returns:
        pd.DataFrame: Daily play counts with columns:
            'date', 'track', 'artist', 'play_count'.
    """
    df = filtered_df.copy()
    # Extract date from timestamp
    df["date"] = df["ts"].dt.date

    # Group by date, track name, and artist
    daily_counts = (
        df.groupby(
            [
                "date",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Rename columns for clarity
    daily_counts.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )

    # Sort by date and descending play count
    return daily_counts.sort_values(["date", "play_count"], ascending=[True, False])

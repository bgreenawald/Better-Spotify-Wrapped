from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.api.api import SpotifyData


def get_most_played_tracks(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the most played tracks from a filtered Spotify listening history DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()

    Returns:
        pd.DataFrame: DataFrame containing tracks sorted by play count with columns:
                     'track_name', 'artist', 'play_count'
    """
    # Group by track and artist, count plays
    play_counts = (
        filtered_df.groupby(
            ["master_metadata_track_name", "master_metadata_album_artist_name"]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Sort by play count in descending order
    sorted_tracks = play_counts.sort_values("play_count", ascending=False)

    # Rename columns for clarity
    sorted_tracks.columns = ["track_name", "artist", "play_count"]

    return sorted_tracks


def get_top_albums(
    filtered_df: pd.DataFrame, spotify_data: SpotifyData
) -> pd.DataFrame:
    """
    Calculate top albums based on median song plays, including songs with zero plays.
    Only considers full albums that appear in the filtered listening history.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()
        spotify_data (SpotifyData): Container with Spotify API data including tracks, artists, and albums

    Returns:
        pd.DataFrame: DataFrame containing albums sorted by median play count with columns:
                     'album_name', 'artist', 'median_plays', 'total_tracks', 'tracks_played'
    """
    # First, identify which albums appear in our listening history
    albums_in_history: Set[str] = set()
    for track_uri in filtered_df["spotify_track_uri"].unique():
        track_id = track_uri.split(":")[-1]
        if track_id in spotify_data.tracks:
            album_id = spotify_data.tracks[track_id]["album"]["id"]
            if spotify_data.albums[album_id]["album_type"] == "album":
                albums_in_history.add(album_id)

    # Create mapping of track URIs to (album_id, track_number) only for relevant albums
    track_to_album: Dict[str, Tuple[str, int]] = {}
    for track_id, track_data in spotify_data.tracks.items():
        album_id = track_data["album"]["id"]
        if album_id not in albums_in_history:
            continue

        track_uri = f"spotify:track:{track_id}"
        # Track numbers are 1-based in the API, convert to 0-based for list indexing
        track_number = track_data["track_number"] - 1
        track_to_album[track_uri] = (album_id, track_number)

    # Count plays per track
    play_counts = filtered_df.groupby("spotify_track_uri").size().to_dict()

    # Initialize only albums from our listening history
    album_tracks: Dict[str, List[int]] = {}
    for album_id in albums_in_history:
        album_tracks[album_id] = [0] * spotify_data.albums[album_id]["total_tracks"]

    # Fill in actual play counts where we have them
    for track_uri, count in play_counts.items():
        if track_uri in track_to_album:
            album_id, track_idx = track_to_album[track_uri]
            # Update the play count at the correct track index
            album_tracks[album_id][track_idx] = count

    # Calculate metrics for each album
    album_stats = []
    for album_id, track_plays in album_tracks.items():
        album_data = spotify_data.albums[album_id]
        album_stats.append(
            {
                "album_name": album_data["name"],
                "artist": album_data["artists"][0]["name"],
                "median_plays": np.median(track_plays),
                "total_tracks": album_data["total_tracks"],
                "tracks_played": sum(1 for plays in track_plays if plays > 0),
                "release_date": album_data["release_date"],
            }
        )

    # Convert to DataFrame and sort by median plays
    results_df = pd.DataFrame(album_stats)
    results_df = results_df.sort_values("median_plays", ascending=False)

    return results_df


def get_top_artist_genres(
    filtered_df: pd.DataFrame,
    spotify_data: SpotifyData,
    unique_tracks: bool = False,
    top_artists_per_genre: int = 2,
) -> pd.DataFrame:
    """
    Calculate the most common genres across unique tracks in the listening history,
    based on the genres of the artists of those tracks. Also includes the top artists
    for each genre based on play count.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()
        spotify_data (SpotifyData): Container with Spotify API data including tracks, artists, and albums
        unique_tracks (bool): If True, only consider unique tracks
        top_artists_per_genre (int): Number of top artists to include per genre

    Returns:
        pd.DataFrame: DataFrame containing genres sorted by frequency with columns:
                     'genre', 'track_count', 'percentage', 'top_artists'
    """
    # Get unique tracks from listening history
    if unique_tracks:
        tracks = filtered_df["spotify_track_uri"].unique()
    else:
        tracks = filtered_df["spotify_track_uri"]

    # Create dictionaries to store genre data
    genre_track_counts = {}  # genre -> track count
    genre_artists = {}  # genre -> {artist_id -> play_count}
    total_tracks = 0

    # For each unique track
    for track_uri in tracks:
        # Extract track ID from URI
        track_id = track_uri.split(":")[-1]

        # Skip if track not in our Spotify data
        if track_id not in spotify_data.tracks:
            continue

        # Get artist ID for this track
        artist_id = spotify_data.tracks[track_id]["artists"][0]["id"]

        # Skip if artist not in our Spotify data
        if artist_id not in spotify_data.artists:
            continue

        # Get artist genres
        artist_genres = spotify_data.artists[artist_id].get("genres", [])

        # If we found genres for this artist
        if artist_genres:
            total_tracks += 1
            # Get artist name
            artist_name = spotify_data.artists[artist_id]["name"]

            # For each genre
            for genre in artist_genres:
                # Increment genre track count
                genre_track_counts[genre] = genre_track_counts.get(genre, 0) + 1

                # Initialize genre's artist dictionary if needed
                if genre not in genre_artists:
                    genre_artists[genre] = {}

                # Increment artist's play count for this genre
                if artist_name not in genre_artists[genre]:
                    genre_artists[genre][artist_name] = 0
                genre_artists[genre][artist_name] += 1

    # Convert to DataFrame with top artists
    genre_df = pd.DataFrame(
        [
            {
                "genre": genre,
                "track_count": count,
                "top_artists": ", ".join(
                    f"{artist} ({plays} plays)"
                    for artist, plays in sorted(
                        genre_artists[genre].items(),
                        key=lambda x: (
                            -x[1],
                            x[0],
                        ),  # Sort by count desc, then name asc
                    )[:top_artists_per_genre]
                ),
            }
            for genre, count in genre_track_counts.items()
        ]
    )

    if len(genre_df) == 0:
        return pd.DataFrame(columns=["genre", "track_count", "percentage"])

    # Calculate percentages
    genre_df["percentage"] = (genre_df["track_count"] / total_tracks * 100).round(2)

    # Sort by track count
    genre_df = genre_df.sort_values("track_count", ascending=False)

    return genre_df


def get_most_played_artists(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the most played artists from a filtered Spotify listening history DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame that has already been filtered using filter_songs()

    Returns:
        pd.DataFrame: DataFrame containing artists sorted by play count with columns:
                     'artist', 'play_count', 'unique_tracks'
    """
    # Count total plays per artist
    play_counts = (
        filtered_df.groupby("master_metadata_album_artist_name")
        .size()
        .reset_index(name="play_count")
    )

    # Count unique tracks per artist
    unique_tracks = (
        filtered_df.groupby("master_metadata_album_artist_name")[
            "master_metadata_track_name"
        ]
        .nunique()
        .reset_index(name="unique_tracks")
    )

    # Merge play counts and unique tracks
    artist_stats = pd.merge(
        play_counts, unique_tracks, on="master_metadata_album_artist_name"
    )

    # Sort by play count in descending order
    sorted_artists = artist_stats.sort_values("play_count", ascending=False)

    # Rename columns for clarity
    sorted_artists.columns = ["artist", "play_count", "unique_tracks"]

    return sorted_artists

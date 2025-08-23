import pandas as pd

from src.api.api import SpotifyData


def add_api_data(
    history_df: pd.DataFrame,
    api_data: SpotifyData,
) -> pd.DataFrame:
    """Add Spotify track, album, artist IDs, and genres to listening history.

    Extracts track ID from the Spotify URI, maps it to album and artist IDs
    from the API data, and adds the artist's genres.

    Args:
        history_df (pd.DataFrame): Listening history with a
            'spotify_track_uri' column.
        api_data (SpotifyData): Spotify API data containing 'tracks' and
            'artists' mappings.

    Returns:
        pd.DataFrame: Updated DataFrame with 'track_id', 'album_id',
            'artist_id', and 'artist_genres' columns added.
    """
    # Extract track ID from the Spotify URI (everything after the last colon)
    history_df["track_id"] = history_df["spotify_track_uri"].str.rsplit(":", n=1).str[-1]

    # Build mapping from track ID to album and artist IDs
    track_to_album = {track_id: info["album"]["id"] for track_id, info in api_data.tracks.items()}
    track_to_artist = {
        track_id: info["artists"][0]["id"] for track_id, info in api_data.tracks.items()
    }

    # Map album_id and artist_id columns
    history_df["album_id"] = history_df["track_id"].map(track_to_album)
    history_df["artist_id"] = history_df["track_id"].map(track_to_artist)

    # Build mapping from artist ID to genres
    artist_to_genres = {
        artist_id: tuple(data["genres"]) for artist_id, data in api_data.artists.items()
    }
    history_df["artist_genres"] = history_df["artist_id"].map(artist_to_genres)

    return history_df


def filter_songs(
    history_df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    exclude_december: bool = True,
    remove_incognito: bool = True,
    excluded_tracks: list[str] | None = None,
    excluded_artists: list[str] | None = None,
    excluded_albums: list[str] | None = None,
    excluded_genres: list[str] | None = None,
) -> pd.DataFrame:
    """Filter listening history based on various criteria.

    Applies filters to exclude podcasts, out-of-range dates, skipped tracks,
    zero playtime, unknown reasons, incognito mode, and optional exclusions
    of specific tracks, artists, albums, or genres.

    Args:
        history_df (pd.DataFrame): Spotify listening history.
        start_date (Optional[pd.Timestamp]): Minimum timestamp (inclusive).
        end_date (Optional[pd.Timestamp]): Maximum timestamp (inclusive).
        exclude_december (bool): Exclude plays from December if True.
        remove_incognito (bool): Exclude incognito-mode plays if True.
        excluded_tracks (Optional[List[str]]): Track names to exclude.
        excluded_artists (Optional[List[str]]): Artist names to exclude.
        excluded_albums (Optional[List[str]]): Album names to exclude.
        excluded_genres (Optional[List[str]]): Genres to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame containing plays that match criteria.
    """
    mask = pd.Series(True, index=history_df.index)

    # Exclude episodes (podcasts) and missing track URIs
    mask &= history_df["episode_name"].isna()
    mask &= history_df["spotify_track_uri"].notna()

    # Apply date range filters
    if start_date is not None:
        mask &= history_df["ts"] >= start_date
    if end_date is not None:
        mask &= history_df["ts"] <= end_date

    # Optionally exclude plays in December
    if exclude_december:
        mask &= history_df["ts"].dt.month < 12

    # Remove skipped tracks and zero-playtime sessions
    mask &= ~history_df["skipped"]
    mask &= history_df["ms_played"] > 0

    # Exclude plays with unknown start or end reasons
    mask &= history_df["reason_start"].ne("unknown")
    mask &= history_df["reason_end"].ne("unknown")

    # Optionally remove incognito-mode plays
    if remove_incognito:
        mask &= ~history_df["incognito_mode"]

    # Optionally exclude specific tracks, artists, albums
    if excluded_tracks:
        mask &= ~history_df["master_metadata_track_name"].isin(excluded_tracks)
    if excluded_artists:
        mask &= ~history_df["master_metadata_album_artist_name"].isin(excluded_artists)
    if excluded_albums:
        mask &= ~history_df["master_metadata_album_album_name"].isin(excluded_albums)

    # Optionally exclude plays by genre
    if excluded_genres:
        mask &= ~history_df["artist_genres"].apply(
            lambda genres: any(genre in excluded_genres for genre in genres) if genres else False
        )

    return history_df.loc[mask].copy()

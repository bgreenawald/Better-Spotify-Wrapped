import pandas as pd


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

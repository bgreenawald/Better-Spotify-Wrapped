"""Module to process Spotify listening history and export top played tracks."""

import os

import duckdb
import pandas as pd

from src.io import load_spotify_history
from src.metrics.metrics import get_most_played_tracks
from src.preprocessing import filter_songs


def main():
    """Process Spotify history and save top songs for each year.

    Loads the full Spotify listening history, filters songs by year,
    and writes the most played tracks to Excel files in the tmp directory
    for years 2023 and 2024.
    """
    # Load the complete listening history once for efficiency
    history_df = load_spotify_history("listening_history")

    # Process each year in the specified range
    con = duckdb.connect(":memory:")
    for year in range(2023, 2025):
        filtered_df = filter_songs(
            history_df,
            start_date=pd.Timestamp(year, 1, 1),
            end_date=pd.Timestamp(year, 12, 31),
            remove_incognito=False,
            exclude_december=False,
        )
        top_songs_df = get_most_played_tracks(filtered_df, con=con)
        output_file = f"tmp/top_songs_{year}.xlsx"
        os.makedirs("tmp", exist_ok=True)
        top_songs_df.to_excel(output_file)


if __name__ == "__main__":
    main()

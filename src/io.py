import json
from pathlib import Path
from typing import List

import pandas as pd


def load_spotify_history(directory: str) -> pd.DataFrame:
    """Load and combine Spotify listening history JSON files into a DataFrame.

    Reads all JSON files in the given directory, each containing a list of
    Spotify listening history records. Combines records into a single
    DataFrame and converts the 'ts' column to timezone-naive datetime.

    Args:
        directory (str): Path to the directory containing .json files
            with Spotify listening history records.

    Returns:
        pd.DataFrame: Combined DataFrame with listening history.
            The 'ts' column is converted to datetime without timezone.
    """
    data_dir = Path(directory)
    all_records: List[dict] = []

    # Load records from each JSON file
    for file_path in data_dir.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as fp:
            records = json.load(fp)
            all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Convert 'ts' column to datetime and remove timezone
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["ts"] = df["ts"].dt.tz_localize(None)

    return df

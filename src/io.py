import json
from pathlib import Path

import pandas as pd


def load_spotify_history(directory: str) -> pd.DataFrame:
    """
    Load all JSON files from a directory containing Spotify listening history
    and combine them into a single pandas DataFrame.

    Args:
        directory (str): Path to directory containing JSON files

    Returns:
        pandas.DataFrame: Combined DataFrame of all listening history
    """
    # Convert directory to Path object
    dir_path = Path(directory)

    # Initialize empty list to store all records
    all_records = []

    # Iterate through all JSON files in directory
    for json_file in dir_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_records.extend(data)

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Convert timestamp to datetime
    df["ts"] = pd.to_datetime(df["ts"])

    df["ts"] = df["ts"].dt.tz_localize(None)

    return df

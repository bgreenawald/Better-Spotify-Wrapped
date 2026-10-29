"""Readers for raw Spotify data-export files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

# Spotify exports audio and video history side by side in one folder. Only the
# audio files belong in listening history; the video ones have a different
# shape and would arrive as mostly-null rows.
AUDIO_GLOB = "*.json"
VIDEO_PREFIX = "streaming_history_video"

# Columns the ingest depends on. Guaranteeing them means an empty or filtered
# directory yields an empty frame with a usable shape rather than no columns.
EXPECTED_COLUMNS = (
    "ts",
    "ms_played",
    "spotify_track_uri",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
    "master_metadata_album_album_name",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
    "conn_country",
    "ip_addr",
)


def load_spotify_history(directory: str) -> pd.DataFrame:
    """Load and combine Spotify listening-history JSON files into a DataFrame.

    Reads the audio history files in `directory`, each a JSON list of play
    records, and returns them as one DataFrame with `ts` parsed as a
    timezone-naive UTC datetime.

    Args:
        directory: Path to the directory holding the export's .json files.

    Returns:
        The combined history. Always carries at least `EXPECTED_COLUMNS`, so an
        empty result still has a usable shape.

    Raises:
        FileNotFoundError: If `directory` does not exist or is not a directory.
        ValueError: If a file does not contain a JSON list of records.
    """
    data_dir = Path(directory)
    if not data_dir.is_dir():
        # Previously this silently produced an empty, column-less frame, so a
        # typo in the path looked like "you have no listening history".
        raise FileNotFoundError(f"History directory not found: {data_dir}")

    # Sorted for deterministic ordering across runs.
    files = sorted(
        f for f in data_dir.glob(AUDIO_GLOB) if not f.name.lower().startswith(VIDEO_PREFIX)
    )
    skipped_video = sorted(
        f.name for f in data_dir.glob(AUDIO_GLOB) if f.name.lower().startswith(VIDEO_PREFIX)
    )
    if skipped_video:
        logging.info(
            "Skipping %d video-history file(s): %s",
            len(skipped_video),
            ", ".join(skipped_video),
        )

    if not files:
        logging.warning("No audio history JSON files found in %s", data_dir)
        return pd.DataFrame({c: pd.Series(dtype="object") for c in EXPECTED_COLUMNS})

    all_records: list[dict] = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                records = json.load(fp)
        except json.JSONDecodeError as e:
            # Name the file: "Expecting value: line 1 column 1" alone is
            # useless when an export holds dozens of them.
            raise ValueError(f"{file_path.name} is not valid JSON: {e}") from e

        if not isinstance(records, list):
            raise ValueError(
                f"{file_path.name} does not contain a JSON list of records "
                f"(got {type(records).__name__})"
            )
        all_records.extend(r for r in records if isinstance(r, dict))
        logging.info("Read %d record(s) from %s", len(records), file_path.name)

    df = pd.DataFrame(all_records)
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    if "ts" in df.columns:
        # utc=True keeps the dtype datetime64 even when offsets are mixed.
        # Without it a mixed export parses to object dtype and the following
        # .dt accessor raises.
        parsed = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        unparsed = int(parsed.isna().sum() - df["ts"].isna().sum())
        if unparsed > 0:
            logging.warning("Could not parse %d timestamp(s); they will be dropped", unparsed)
        df["ts"] = parsed.dt.tz_localize(None)

    return df

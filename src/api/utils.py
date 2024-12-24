#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import click
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
API_DATA_CACHE = Path("data/api/cache")


def extract_ids_from_tracks(tracks_file: str) -> Dict[str, Set[str]]:
    """
    Parse track JSON files and extract all unique artist and album IDs.

    Args:
        tracks_file: Path to list of track IDs

    Returns:
        Dictionary containing sets of artist_ids and album_ids
    """
    artist_ids = set()
    album_ids = set()

    # Ensure directory exists
    if not Path(tracks_file).exists():
        raise ValueError(f"File not found: {tracks_file}")

    # Get all JSON files in directory
    with open(tracks_file, "r") as f:
        track_files = f.readlines()

    track_files = [f"{file.strip()}.json" for file in track_files]

    for track_file in track_files:
        try:
            with open(API_DATA_CACHE / "tracks" / track_file, "r") as f:
                track_data = json.load(f)

            # Extract artist IDs
            for artist in track_data["artists"]:
                artist_ids.add(artist["id"])

            # Extract album ID
            album_ids.add(track_data["album"]["id"])

        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping malformed JSON file: {track_file}")
        except KeyError as e:
            click.echo(f"Warning: Missing expected field {e} in file: {track_file}")
        except Exception as e:
            click.echo(f"Warning: Unexpected error processing {track_file}: {e}")

    return {"artist_ids": artist_ids, "album_ids": album_ids}


def extract_track_ids_from_history(history_dir: Path) -> List[str]:
    """
    Parse streaming history JSON files and extract all track IDs.

    Args:
        history_dir: Path to directory containing streaming history JSON files

    Returns:
        List of track IDs (without spotify:track prefix)
    """
    track_ids = set()

    # Get all JSON files in directory
    history_files = history_dir.glob("*.json")

    for history_file in history_files:
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)

            # Process each record in the file
            for record in history_data:
                track_uri = record.get("spotify_track_uri")
                if track_uri and track_uri.startswith("spotify:track:"):
                    # Remove the spotify:track: prefix and add to set
                    track_id = track_uri.split(":")[-1]
                    track_ids.add(track_id)

        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping malformed JSON file: {history_file}")
        except Exception as e:
            click.echo(f"Warning: Unexpected error processing {history_file}: {e}")

    return sorted(list(track_ids))


def save_ids_to_file(ids: Set[str], output_file: Path):
    """
    Save a set of IDs to a text file, one ID per line.

    Args:
        ids: Set of IDs to save
        output_file: Path where to save the file
    """
    with open(output_file, "w") as f:
        for item_id in sorted(ids):
            f.write(f"{item_id}\n")


@click.group()
def cli():
    """Utility commands for processing Spotify data files."""
    pass


@cli.command()
@click.argument(
    "tracks_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--artists-output",
    type=click.Path(path_type=Path),
    default=Path(DATA_DIR) / "artist_ids.txt" if DATA_DIR else Path("artist_ids.txt"),
    help="Output file for artist IDs",
)
@click.option(
    "--albums-output",
    type=click.Path(path_type=Path),
    default=Path(DATA_DIR) / "album_ids.txt" if DATA_DIR else Path("album_ids.txt"),
    help="Output file for album IDs",
)
def extract(tracks_file: Path, artists_output: Path, albums_output: Path):
    """Extract artist and album IDs from track JSON files."""
    click.echo(f"Processing track files from {tracks_file}...")

    # Extract IDs
    ids = extract_ids_from_tracks(tracks_file)

    # Save to files
    save_ids_to_file(ids["artist_ids"], artists_output)
    save_ids_to_file(ids["album_ids"], albums_output)

    click.echo(f"\nExtracted {len(ids['artist_ids'])} unique artist IDs")
    click.echo(f"Extracted {len(ids['album_ids'])} unique album IDs")
    click.echo(f"\nArtist IDs saved to: {artists_output}")
    click.echo(f"Album IDs saved to: {albums_output}")


@cli.command()
@click.argument(
    "history_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path(DATA_DIR) / "track_ids.txt" if DATA_DIR else Path("track_ids.txt"),
    help="Output file for track IDs",
)
def extract_history(history_dir: Path, output: Path):
    """Extract track IDs from streaming history JSON files."""
    click.echo(f"Processing streaming history files from {history_dir}...")

    # Extract track IDs
    track_ids = extract_track_ids_from_history(history_dir)

    # Save to file
    with open(output, "w") as f:
        for track_id in track_ids:
            f.write(f"{track_id}\n")

    click.echo(f"\nExtracted {len(track_ids)} unique track IDs")
    click.echo(f"Track IDs saved to: {output}")


if __name__ == "__main__":
    cli()

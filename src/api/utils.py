#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, Set

import click
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")


def extract_ids_from_tracks(tracks_cache_dir: Path) -> Dict[str, Set[str]]:
    """
    Parse track JSON files and extract all unique artist and album IDs.

    Args:
        tracks_cache_dir: Path to directory containing track JSON files

    Returns:
        Dictionary containing sets of artist_ids and album_ids
    """
    artist_ids = set()
    album_ids = set()

    # Ensure directory exists
    if not tracks_cache_dir.exists():
        raise ValueError(f"Directory not found: {tracks_cache_dir}")

    # Get all JSON files in directory
    track_files = tracks_cache_dir.glob("*.json")

    for track_file in track_files:
        try:
            with open(track_file, "r") as f:
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
    """Extract artist and album IDs from Spotify track cache."""
    pass


@cli.command()
@click.argument(
    "tracks_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
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
def extract(tracks_dir: Path, artists_output: Path, albums_output: Path):
    """Extract artist and album IDs from track JSON files."""
    click.echo(f"Processing track files from {tracks_dir}...")

    # Extract IDs
    ids = extract_ids_from_tracks(tracks_dir)

    # Save to files
    save_ids_to_file(ids["artist_ids"], artists_output)
    save_ids_to_file(ids["album_ids"], albums_output)

    click.echo(f"\nExtracted {len(ids['artist_ids'])} unique artist IDs")
    click.echo(f"Extracted {len(ids['album_ids'])} unique album IDs")
    click.echo(f"\nArtist IDs saved to: {artists_output}")
    click.echo(f"Album IDs saved to: {albums_output}")


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""Command-line utilities to extract Spotify data IDs."""

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR = os.getenv("DATA_DIR")
API_DATA_CACHE = Path("data/api/cache")


def extract_ids_from_tracks(tracks_file: Path) -> dict[str, set[str]]:
    """Extract unique artist and album IDs from track JSON files.

    Reads a text file listing track ID stems, loads each corresponding JSON
    file from the API data cache, and collects all unique artist and album
    IDs.

    Args:
        tracks_file (Path): Path to a text file containing track ID stems,
            one per line.

    Returns:
        Dict[str, Set[str]]: Dictionary with keys 'artist_ids' and 'album_ids'.

    Raises:
        FileNotFoundError: If the tracks file does not exist.
    """
    tracks_file = Path(tracks_file)
    if not tracks_file.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_file}")

    # Read track ID stems from file, ignoring blank lines
    stems = tracks_file.read_text(encoding="utf-8").splitlines()
    track_files = [f"{stem.strip()}.json" for stem in stems if stem.strip()]

    artist_ids: set[str] = set()
    album_ids: set[str] = set()

    for filename in track_files:
        track_path = API_DATA_CACHE / "tracks" / filename
        try:
            with track_path.open(encoding="utf-8") as fp:
                data = json.load(fp)
            # Collect artist IDs
            for artist in data["artists"]:
                artist_ids.add(artist["id"])
            # Collect album ID
            album_ids.add(data["album"]["id"])
        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping malformed JSON file: {filename}")
        except KeyError as e:
            click.echo(f"Warning: Missing expected field {e} in file: {filename}")
        except Exception as e:
            click.echo(f"Warning: Unexpected error processing {filename}: {e}")

    return {"artist_ids": artist_ids, "album_ids": album_ids}


def extract_track_ids_from_history(history_dir: Path) -> set[str]:
    """Extract unique track IDs from streaming history JSON files.

    Parses all JSON files in the specified directory, extracts the track ID
    from the 'spotify_track_uri' field (removing the 'spotify:track:' prefix),
    and returns a set of unique IDs.

    Args:
        history_dir (Path): Directory containing streaming history JSON files.

    Returns:
        Set[str]: Unique Spotify track IDs without the URI prefix.
    """
    track_ids: set[str] = set()

    for history_file in history_dir.glob("*.json"):
        try:
            with history_file.open(encoding="utf-8") as fp:
                records = json.load(fp)
            for record in records:
                uri = record.get("spotify_track_uri", "")
                if uri.startswith("spotify:track:"):
                    # Extract ID after the last colon
                    track_id = uri.split(":", maxsplit=2)[-1]
                    track_ids.add(track_id)
        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping malformed JSON file: {history_file.name}")
        except Exception as e:
            click.echo(f"Warning: Unexpected error processing {history_file.name}: {e}")

    return track_ids


def save_ids_to_file(ids: set[str], output_file: Path) -> None:
    """Save a set of IDs to a text file, one ID per line.

    The IDs are written in sorted order.

    Args:
        ids (Set[str]): IDs to write.
        output_file (Path): Path to the output text file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fp:
        for item in sorted(ids):
            fp.write(f"{item}\n")


@click.group()
def cli() -> None:
    """Spotify data processing command group."""
    pass


@cli.command()
@click.argument(
    "tracks_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--artists-output",
    type=click.Path(path_type=Path),
    default=(Path(DATA_DIR) / "artist_ids.txt" if DATA_DIR else Path("artist_ids.txt")),
    help="Output file for artist IDs.",
)
@click.option(
    "--albums-output",
    type=click.Path(path_type=Path),
    default=(Path(DATA_DIR) / "album_ids.txt" if DATA_DIR else Path("album_ids.txt")),
    help="Output file for album IDs.",
)
def extract(tracks_file: Path, artists_output: Path, albums_output: Path) -> None:
    """Extract artist and album IDs from track JSON files.

    Reads a list of track ID stems from the given file and writes unique
    artist and album IDs to the specified output files.
    """
    click.echo(f"Processing track list: {tracks_file}...")

    ids = extract_ids_from_tracks(tracks_file)
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
    default=(Path(DATA_DIR) / "track_ids.txt" if DATA_DIR else Path("track_ids.txt")),
    help="Output file for track IDs.",
)
def extract_history(history_dir: Path, output: Path) -> None:
    """Extract track IDs from streaming history JSON files.

    Parses JSON files in the specified directory and writes unique track IDs
    (without the 'spotify:track:' prefix) to the given output file.
    """
    click.echo(f"Processing streaming history: {history_dir}...")

    track_ids = extract_track_ids_from_history(history_dir)
    save_ids_to_file(track_ids, output)

    click.echo(f"\nExtracted {len(track_ids)} unique track IDs")
    click.echo(f"Track IDs saved to: {output}")


if __name__ == "__main__":
    cli()

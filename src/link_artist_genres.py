"""Link artist genres from cached Spotify artist JSONs into the database.

This script scans `data/api/cache/artists/*.json` (or a provided cache dir),
reads each artist payload and links canonical genres into `artist_genres`
using mappings in `map_genre` (provenance `source='spotify'`).

It reuses `populate_artist_genres_from_cache` from `src.api.api`.

Usage examples:

  python -m src.link_artist_genres --db data/db/music.db
  python -m src.link_artist_genres --db refactor-layout/data/db/music.db --cache-dir data/api/cache

Ensure the schema from `DDL.sql` has been applied so that `artist_genres`,
`dim_artists`, `dim_genres`, and `map_genre` exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .api.api import populate_artist_genres_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Link artists to canonical genres in artist_genres from cached Spotify artist JSONs."
        )
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        type=Path,
        default=Path("data/db/music.db"),
        help="Path to DuckDB database file (with DDL applied).",
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        type=Path,
        default=None,
        help=(
            "Override cache directory root (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache)."
        ),
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Optional cap on number of artist JSON files to process.",
    )
    args = parser.parse_args()

    counts = populate_artist_genres_from_cache(
        db_path=args.db_path,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        limit=args.limit,
    )
    print(
        " | ".join(
            [
                f"Artists scanned: {counts['artists_scanned']}",
                f"Links inserted: {counts['links_inserted']}",
                f"Unmapped tags: {counts['unmapped_tags']}",
                f"Artists missing in dim_artists: {counts['artists_missing_dim']}",
            ]
        )
    )


if __name__ == "__main__":
    main()

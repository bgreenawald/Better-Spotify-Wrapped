from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

# Parent genres are specified as a constant here (level 0 taxonomy)
PARENT_GENRES: list[str] = [
    "blues",
    "children's music",
    "classical",
    "comedy",
    "country",
    "edm",
    "electronica",
    "folk",
    "funk",
    "gospel",
    "hip hop",
    "jazz",
    "metal",
    "new age",
    "opera",
    "pop",
    "r&b",
    "rap",
    "reggae",
    "rock",
    "soul",
    "soundtrack",
    "world",
]


def _slugify(name: str) -> str:
    """Create a slug by replacing spaces with underscores only.

    Keeps punctuation as-is per requirement.
    """
    return name.replace(" ", "_")


def _display_name(name: str) -> str:
    """Capitalize genre for display in `dim_genres.name`.

    Use simple sentence-style capitalization to avoid odd Title-Case like "Children'S".
    """
    s = name.strip()
    return s[:1].upper() + s[1:] if s else s


def _load_mapping_file(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize keys and values to lowercase/stripped for matching
    norm: dict[str, list[str]] = {}
    for child, parents in data.items():
        c = str(child).strip().lower()
        ps = [str(p).strip().lower() for p in (parents or [])]
        norm[c] = ps
    return norm


def _ensure_dim_genres(
    conn: duckdb.DuckDBPyConnection, *, canonical_genres: Iterable[str], level: int = 0
) -> dict[str, int]:
    """Ensure each canonical genre exists in `dim_genres` and return name→genre_id map.

    Args:
        conn: DuckDB connection.
        canonical_genres: iterable of lowercase canonical genre names.
        level: integer level to store (0 for parents).

    Returns:
        dict mapping lowercase canonical name to genre_id in `dim_genres`.
    """
    canonical = sorted({g.strip().lower() for g in canonical_genres if g and g.strip()})
    if not canonical:
        return {}

    df = pd.DataFrame(
        {
            "slug": [_slugify(g) for g in canonical],
            "name": [_display_name(g) for g in canonical],
            "level": level,
            "active": True,
            # Keep original lowercase name for lookup join later
            "_name_lc": canonical,
        }
    )

    conn.register("df_canonical", df)
    # Insert any missing canonical rows using anti-join on slug
    conn.execute(
        """
        INSERT INTO dim_genres (slug, name, level, active)
        SELECT c.slug, c.name, c.level, c.active
        FROM df_canonical c
        ANTI JOIN dim_genres d ON d.slug = c.slug
        """
    )

    # Fetch genre_id for all provided canonical names
    rows = conn.execute(
        """
        SELECT d.genre_id, c._name_lc AS name_lc
        FROM df_canonical c
        JOIN dim_genres d ON d.slug = c.slug
        """
    ).fetchall()
    return {name_lc: int(genre_id) for (genre_id, name_lc) in rows}


def _prepare_mappings(
    *,
    child_to_parents: dict[str, list[str]],
    parent_list: list[str],
) -> list[tuple[str, str]]:
    """Build (tag_raw, canonical_parent) rows from mapping.

    Rules:
    - If child equals a parent (case-insensitive), map it to itself.
    - Else if any parents in file intersect the provided parent_list, map to the
      first match in the order of `parent_list`.
    - Else (no mapping), map to 'other'.
    """
    parents_lc = [p.strip().lower() for p in parent_list]

    rows: list[tuple[str, str]] = []
    for child_raw, mapped_parents in child_to_parents.items():
        child_lc = child_raw.strip().lower()

        if child_lc in parents_lc:
            # Child is itself a parent → map to itself (avoid 'other')
            rows.append((child_raw, child_lc))
            continue

        # Intersection in the order of provided parents list
        mapped_set = {p.strip().lower() for p in mapped_parents or []}
        chosen: str | None = None
        for p in parents_lc:
            if p in mapped_set:
                chosen = p
                break

        if not chosen:
            chosen = "other"

        rows.append((child_raw, chosen))

    # Also ensure parents themselves map to themselves (useful for normalization)
    for p in parents_lc:
        rows.append((p, p))

    # Always include 'other' self-mapping as a catch-all
    rows.append(("other", "other"))

    # Deduplicate rows while preserving first occurrence
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for r in rows:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def _choose_parent_for_children(
    *, child_to_parents: dict[str, list[str]], parent_list: list[str]
) -> dict[str, str]:
    """Choose a single canonical parent for each child.

    Returns a dict mapping child_raw → chosen_parent (lowercase).
    If child equals a parent, parent is itself. If no match, 'other'.
    """
    parents_lc = [p.strip().lower() for p in parent_list]
    out: dict[str, str] = {}
    for child_raw, mapped_parents in child_to_parents.items():
        child_lc = child_raw.strip().lower()
        if child_lc in parents_lc:
            out[child_raw] = child_lc
            continue
        mapped_set = {p.strip().lower() for p in (mapped_parents or [])}
        chosen = next((p for p in parents_lc if p in mapped_set), None)
        out[child_raw] = chosen or "other"
    return out


def load_genre_taxonomy(
    *,
    db_path: str | Path,
    mapping_file: str | Path,
    parents: list[str] | None = None,
    source: str = "spotify",
) -> dict[str, int]:
    """Load taxonomy into `dim_genres` and `map_genre`.

    Returns summary counts dict.
    """
    db_path = str(db_path)
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    # Normalize parent list (use module constant if not provided)
    if parents is None:
        parents = PARENT_GENRES
    parents_lc = [p.strip().lower() for p in parents if p and p.strip()]
    if not parents_lc:
        raise ValueError("At least one parent genre must be provided")

    # Load mapping JSON
    child_to_parents = _load_mapping_file(mapping_path)

    # Connect and start transaction
    conn = duckdb.connect(db_path)
    try:
        conn.execute("BEGIN TRANSACTION;")

        # Ensure canonical parents + 'other' exist in dim_genres
        canonical_map = _ensure_dim_genres(conn, canonical_genres=[*parents_lc, "other"], level=0)

        # Determine chosen parent for each child
        child_parent = _choose_parent_for_children(
            child_to_parents=child_to_parents, parent_list=parents_lc
        )

        # Insert level-1 subgenres into dim_genres (exclude children that are parents)
        children = [c for c in child_parent if c.strip().lower() not in parents_lc]
        inserted_children = 0
        if children:
            child_df = pd.DataFrame(
                {
                    "slug": [_slugify(c) for c in children],
                    "name": [_display_name(c) for c in children],
                    "level": 1,
                    "active": True,
                }
            )
            conn.register("df_children", child_df)
            # Count pending inserts via anti-join
            inserted_children = conn.execute(
                """
                SELECT COUNT(*)
                FROM df_children c
                ANTI JOIN dim_genres d ON d.slug = c.slug
                """
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO dim_genres (slug, name, level, active)
                SELECT c.slug, c.name, c.level, c.active
                FROM df_children c
                ANTI JOIN dim_genres d ON d.slug = c.slug
                """
            )

        # Build map_genre rows (child tag → chosen canonical parent)
        mappings = [(child, parent) for child, parent in child_parent.items()]
        # add parent/self and other/self rows for convenience
        mappings.extend([(p, p) for p in parents_lc])
        mappings.append(("other", "other"))
        # Dedup
        seen = set()
        dedup = []
        for pair in mappings:
            if pair not in seen:
                seen.add(pair)
                dedup.append(pair)
        map_df = pd.DataFrame(dedup, columns=["tag_raw", "canonical_parent"])
        # Join to genre_id via canonical parent
        id_map_df = pd.DataFrame(
            {
                "canonical_parent": list(canonical_map.keys()),
                "genre_id": list(canonical_map.values()),
            }
        )
        map_df = map_df.merge(id_map_df, on="canonical_parent", how="left")
        map_df["source"] = source
        map_df["confidence"] = 1.0

        # Check for resolution failures
        unresolved = map_df[map_df["genre_id"].isna()]
        if not unresolved.empty:
            unresolved_parents = unresolved["canonical_parent"].unique().tolist()
            raise ValueError(
                f"Failed to resolve genre_id for canonical parents: {unresolved_parents}. "
                f"Ensure these exist in dim_genres."
            )
        map_df = map_df.dropna(subset=["genre_id"]).copy()
        map_df["genre_id"] = map_df["genre_id"].astype(int)

        # Insert into map_genre using anti-join on (source, tag_raw)
        conn.register("df_map", map_df[["source", "tag_raw", "genre_id", "confidence"]])
        inserted_map = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_map m
            ANTI JOIN map_genre g
              ON g.source = m.source AND g.tag_raw = m.tag_raw
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO map_genre (source, tag_raw, genre_id, confidence)
            SELECT m.source, m.tag_raw, m.genre_id, m.confidence
            FROM df_map m
            ANTI JOIN map_genre g
              ON g.source = m.source AND g.tag_raw = m.tag_raw
            """
        )

        # Insert genre_hierarchy entries (parent → child) for level-1 children
        inserted_hierarchy = 0
        if children:
            # fetch child ids
            child_ids = conn.execute(
                """
                SELECT d.genre_id, d.slug
                FROM dim_genres d
                WHERE d.slug IN (
                    SELECT slug FROM df_children
                )
                """
            ).fetchall()
            child_slug_to_id = {slug: int(gid) for (gid, slug) in child_ids}

            # Build hierarchy pairs using chosen parent
            hier_pairs: list[tuple[int, int]] = []
            for c in children:
                parent = child_parent[c]
                # Skip if somehow parent unresolved
                p_id = canonical_map.get(parent)
                c_id = child_slug_to_id.get(_slugify(c))
                if p_id is None or c_id is None:
                    continue
                # Skip if child equals parent (should already be excluded)
                hier_pairs.append((p_id, c_id))

            if hier_pairs:
                hier_df = pd.DataFrame(hier_pairs, columns=["parent_genre_id", "child_genre_id"])
                conn.register("df_hier", hier_df)
                # Count pending inserts via anti-join
                inserted_hierarchy = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM df_hier h
                    ANTI JOIN genre_hierarchy g
                      ON g.parent_genre_id = h.parent_genre_id AND g.child_genre_id = h.child_genre_id
                    """
                ).fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO genre_hierarchy (parent_genre_id, child_genre_id)
                    SELECT h.parent_genre_id, h.child_genre_id
                    FROM df_hier h
                    ANTI JOIN genre_hierarchy g
                      ON g.parent_genre_id = h.parent_genre_id AND g.child_genre_id = h.child_genre_id
                    """
                )

        conn.execute("COMMIT;")
        return {
            "inserted_dim_genres": len(canonical_map),
            "inserted_map_genre": int(inserted_map),
            "inserted_subgenres_level1": int(inserted_children),
            "inserted_genre_hierarchy": int(inserted_hierarchy),
        }
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load subgenre taxonomy into dim_genres (levels 0 and 1), map_genre, and genre_hierarchy."
        )
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        type=str,
        default="data/db/music.db",
        help="Path to DuckDB database (default: data/db/music.db)",
    )
    parser.add_argument(
        "--json",
        dest="mapping_file",
        type=str,
        required=True,
        help="Path to JSON mapping file (child → [parents])",
    )
    # Parents are specified via PARENT_GENRES constant in this module
    parser.add_argument(
        "--source",
        dest="source",
        type=str,
        default="spotify",
        help="Source string for map_genre (default: spotify)",
    )

    args = parser.parse_args()
    summary = load_genre_taxonomy(
        db_path=args.db_path,
        mapping_file=args.mapping_file,
        source=args.source,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "db": str(args.db_path),
                "json": str(args.mapping_file),
                "inserted_dim_genres": summary["inserted_dim_genres"],
                "inserted_map_genre": summary["inserted_map_genre"],
                "inserted_subgenres_level1": summary["inserted_subgenres_level1"],
                "inserted_genre_hierarchy": summary["inserted_genre_hierarchy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

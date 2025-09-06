from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Mode = Literal["tracks", "artists", "genres"]


@dataclass(frozen=True)
class RegionItem:
    id: str
    name: str
    joint_rank: float
    ranks: dict[str, int]
    counts: dict[str, int]


def _register_list_param(con, name: str, values: Iterable[str] | None) -> str:
    """Register a small list as a temp relation and return its table name."""
    if not values:
        return ""
    df = pd.DataFrame({"val": list(values)})
    relname = f"tmp_{name}"
    with contextlib.suppress(Exception):
        con.unregister(relname)
    con.register(relname, df)
    return relname


def _base_plays_sql(
    *,
    users_rel: str,
    exclude_december: bool,
    remove_incognito: bool,
    rel_tracks: str,
    rel_artists: str,
    rel_albums: str,
    rel_genres: str,
) -> str:
    # Build optional WHERE fragments
    filters = [
        f"p.user_id IN (SELECT val FROM {users_rel})",
        "COALESCE(p.skipped, FALSE) = FALSE",
        "COALESCE(p.duration_ms, 0) > 0",
        "COALESCE(p.reason_start, 'unknown') <> 'unknown'",
        "COALESCE(p.reason_end, 'unknown') <> 'unknown'",
    ]
    if exclude_december:
        filters.append("EXTRACT(MONTH FROM p.played_at) <> 12")
    if remove_incognito:
        filters.append("COALESCE(p.incognito_mode, FALSE) = FALSE")

    extra_exclusions: list[str] = []
    if rel_tracks:
        extra_exclusions.append(f"COALESCE(t.track_name, '') NOT IN (SELECT val FROM {rel_tracks})")
    if rel_artists:
        extra_exclusions.append(
            f"COALESCE(ar.artist_name, '') NOT IN (SELECT val FROM {rel_artists})"
        )
    if rel_albums:
        extra_exclusions.append(
            f"COALESCE(al.album_name, '') NOT IN (SELECT val FROM {rel_albums})"
        )
    # Genre exclusions (by name) via artist_genres + genre_hierarchy when present
    if rel_genres:
        extra_exclusions.append(
            "NOT EXISTS (\n"
            "  SELECT 1\n"
            "  FROM bridge_track_artists b\n"
            "  JOIN artist_genres ag ON ag.artist_id = b.artist_id\n"
            "  JOIN dim_genres g ON g.genre_id = ag.genre_id\n"
            "  LEFT JOIN genre_hierarchy gh ON gh.child_genre_id = g.genre_id\n"
            "  LEFT JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id\n"
            "  WHERE b.role = 'primary' AND b.track_id = p.track_id\n"
            f"    AND (lower(g.name) IN (SELECT lower(val) FROM {rel_genres})\n"
            f"         OR lower(COALESCE(pg.name, '')) IN (SELECT lower(val) FROM {rel_genres}))\n"
            ")"
        )

    where_clause = " AND\n          ".join(filters + extra_exclusions)

    # Build JOINs, including albums only if album exclusions are provided
    joins = [
        "LEFT JOIN dim_tracks t ON t.track_id = p.track_id",
        "LEFT JOIN bridge_track_artists b ON b.track_id = p.track_id AND b.role = 'primary'",
        "LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id",
    ]
    if rel_albums:
        joins.append("LEFT JOIN dim_albums al ON al.album_id = t.album_id")

    joins_sql = "\n        ".join(joins)

    return (
        "WITH base AS (\n"
        "  SELECT p.user_id, p.track_id, p.played_at\n"
        "  FROM fact_plays p\n"
        f"  {joins_sql}\n"
        f"  WHERE {where_clause}\n"
        ")\n"
    )


def _ranks_sql(mode: Mode, *, hide_parent_genres: bool = False) -> str:
    """Return the CTEs for per-user counts and ranks, prefixed with a comma.

    Must follow immediately after "WITH base AS (...)".
    """
    if mode == "tracks":
        return "".join(
            [
                ",\n",
                "counts AS (\n",
                "  SELECT b.user_id, t.track_id AS entity_id, t.track_name AS entity_name, COUNT(*) AS play_count\n",
                "  FROM base b\n",
                "  JOIN dim_tracks t ON t.track_id = b.track_id\n",
                "  GROUP BY 1,2,3\n",
                "),\n",
                "ranks AS (\n",
                "  SELECT user_id, entity_id, entity_name, play_count,\n",
                "         DENSE_RANK() OVER (PARTITION BY user_id ORDER BY play_count DESC, entity_name ASC) AS rnk\n",
                "  FROM counts\n",
                ")\n",
            ]
        )
    if mode == "artists":
        return "".join(
            [
                ",\n",
                "track_primary AS (\n",
                "  SELECT b.user_id, b.track_id, a.artist_id, ar.artist_name\n",
                "  FROM base b\n",
                "  JOIN bridge_track_artists a ON a.track_id = b.track_id AND a.role = 'primary'\n",
                "  JOIN dim_artists ar ON ar.artist_id = a.artist_id\n",
                "),\n",
                "counts AS (\n",
                "  SELECT user_id, artist_id AS entity_id, artist_name AS entity_name, COUNT(*) AS play_count\n",
                "  FROM track_primary\n",
                "  GROUP BY 1,2,3\n",
                "),\n",
                "ranks AS (\n",
                "  SELECT user_id, entity_id, entity_name, play_count,\n",
                "         DENSE_RANK() OVER (PARTITION BY user_id ORDER BY play_count DESC, entity_name ASC) AS rnk\n",
                "  FROM counts\n",
                ")\n",
            ]
        )
    # genres
    cond = " AND COALESCE(dg.level, 1) <> 0" if hide_parent_genres else ""
    return "".join(
        [
            ",\n",
            "track_primary AS (\n",
            "  SELECT b.user_id, b.track_id, a.artist_id\n",
            "  FROM base b\n",
            "  JOIN bridge_track_artists a ON a.track_id = b.track_id AND a.role = 'primary'\n",
            "),\n",
            "artist_genre AS (\n",
            "  SELECT tp.user_id, g.genre_id AS entity_id, dg.name AS entity_name\n",
            "  FROM track_primary tp\n",
            "  JOIN artist_genres g ON g.artist_id = tp.artist_id\n",
            "  JOIN dim_genres dg ON dg.genre_id = g.genre_id",
            cond,
            "\n",
            "),\n",
            "counts AS (\n",
            "  SELECT user_id, entity_id, entity_name, COUNT(*) AS play_count\n",
            "  FROM artist_genre\n",
            "  GROUP BY 1,2,3\n",
            "),\n",
            "ranks AS (\n",
            "  SELECT user_id, entity_id, entity_name, play_count,\n",
            "         DENSE_RANK() OVER (PARTITION BY user_id ORDER BY play_count DESC, entity_name ASC) AS rnk\n",
            "  FROM counts\n",
            ")\n",
        ]
    )


def _build_users_rel(con, users: Iterable[str]) -> str:
    df = pd.DataFrame({"val": list(users)})
    rel = "tmp_users"
    with contextlib.suppress(Exception):
        con.unregister(rel)
    con.register(rel, df)
    return rel


def compute_social_regions(
    *,
    con: Any,
    users: list[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    mode: Mode = "tracks",
    exclude_december: bool = True,
    remove_incognito: bool = True,
    excluded_tracks: Iterable[str] | None = None,
    excluded_artists: Iterable[str] | None = None,
    excluded_albums: Iterable[str] | None = None,
    excluded_genres: Iterable[str] | None = None,
    hide_parent_genres: bool = False,
    limit_per_region: int = 10,
) -> dict:
    """Compute Venn regions for 2–3 users with per-user ranks and joint ranks.

    Returns a dict with keys: users, mode, regions -> mapping of region_key to data.
    Region keys for 2 users: "U1_only", "U2_only", "U1_U2". For 3 users, adds
    "U1_only", "U2_only", "U3_only", "U1_U2", "U1_U3", "U2_U3", "U1_U2_U3".
    Keys include the actual user_id values in order.
    """
    if not users or len(users) < 2 or len(users) > 3:
        raise ValueError("users must contain 2 or 3 user ids")

    # Prepare parameter relations
    users_rel = _build_users_rel(con, users)
    rel_tracks = _register_list_param(con, "excluded_tracks", excluded_tracks)
    rel_artists = _register_list_param(con, "excluded_artists", excluded_artists)
    rel_albums = _register_list_param(con, "excluded_albums", excluded_albums)
    rel_genres = _register_list_param(con, "excluded_genres", excluded_genres)

    # Resolve display names for selected users
    try:
        labels_sql = f"SELECT val as user_id, COALESCE(d.display_name, val) AS label FROM {users_rel} u LEFT JOIN dim_users d ON d.user_id = u.val"
        labels_df = con.execute(labels_sql).df()
        user_labels = {str(r.user_id): str(r.label) for r in labels_df.itertuples(index=False)}
    except Exception:
        user_labels = {str(u): str(u) for u in users}

    # Register date bounds if provided
    params: list[Any] = []
    date_clause = ""
    if start is not None:
        date_clause += " AND p.played_at >= ?"
        params.append(pd.to_datetime(start).to_pydatetime())
    if end is not None:
        date_clause += " AND p.played_at <= ?"
        params.append(pd.to_datetime(end).to_pydatetime())

    base = _base_plays_sql(
        users_rel=users_rel,
        exclude_december=exclude_december,
        remove_incognito=remove_incognito,
        rel_tracks=rel_tracks,
        rel_artists=rel_artists,
        rel_albums=rel_albums,
        rel_genres=rel_genres,
    ).replace("WHERE ", f"WHERE 1=1{date_clause} AND ")

    ranks = _ranks_sql(mode, hide_parent_genres=hide_parent_genres)

    # Per-user play counts in the filtered base
    counts_sql = base + "SELECT user_id, COUNT(*) AS plays FROM base GROUP BY 1;"
    user_counts_df = con.execute(counts_sql, params).df()
    user_counts = {str(r.user_id): int(r.plays) for r in user_counts_df.itertuples(index=False)}

    sql = base + ranks + "SELECT user_id, entity_id, entity_name, play_count, rnk FROM ranks;"

    df = con.execute(sql, params).df()
    if df.empty:
        return {
            "users": users,
            "mode": mode,
            "regions": {},
            "totals": {},
            "user_counts": user_counts,
            "user_labels": user_labels,
        }

    # Build per-user rankings and sets
    per_user = {u: df[df.user_id == u].copy() for u in users}
    for u in users:
        per_user[u].sort_values(["rnk", "entity_name"], inplace=True)

    # Helper to collect items for a set of users
    def _region_entities(required: tuple[str, ...], excluded: tuple[str, ...]) -> pd.DataFrame:
        req_sets = []
        for u in required:
            req_sets.append(set(per_user[u]["entity_id"].unique().tolist()))
        if not req_sets:
            return pd.DataFrame(columns=df.columns)
        entities = set.intersection(*req_sets)
        for u in excluded:
            entities -= set(per_user[u]["entity_id"].unique().tolist())
        if not entities:
            return pd.DataFrame(columns=df.columns)
        subset = df[df["entity_id"].isin(list(entities)) & df["user_id"].isin(list(required))]
        # Compute joint rank
        agg = (
            subset.groupby(["entity_id", "entity_name"], as_index=False)
            .agg(joint_rank=("rnk", "mean"))
            .sort_values(["joint_rank", "entity_name"])
        )
        return agg

    regions: dict[str, list[RegionItem]] = {}
    totals: dict[str, int] = {}

    # Build region keys using actual user ids for clarity
    if len(users) == 2:
        u1, u2 = users
        # Intersection
        inter = _region_entities((u1, u2), ())
        totals[f"{u1}_{u2}"] = int(inter.shape[0])
        top = inter.head(limit_per_region)
        items: list[RegionItem] = []
        for row in top.itertuples(index=False):
            eid = str(row.entity_id)
            nm = str(row.entity_name)
            # gather counts/ranks per user
            ranks_counts: dict[str, tuple[int, int]] = {}
            for u in (u1, u2):
                r = per_user[u]
                rr = r[r.entity_id == row.entity_id].sort_values(["rnk", "entity_name"]).iloc[0]
                ranks_counts[u] = (int(rr.rnk), int(rr.play_count))
            items.append(
                RegionItem(
                    id=eid,
                    name=nm,
                    joint_rank=float(row.joint_rank),
                    ranks={k: v[0] for k, v in ranks_counts.items()},
                    counts={k: v[1] for k, v in ranks_counts.items()},
                )
            )
        regions[f"{u1}_{u2}"] = [item.__dict__ for item in items]

        # Non-intersections: top per user overall (duplicates with intersection allowed)
        for u in (u1, u2):
            ru = per_user[u]
            totals[f"{u}_only"] = int(ru["entity_id"].nunique())
            ru_sorted = ru.sort_values(["rnk", "entity_name"]).head(limit_per_region)
            u_items: list[RegionItem] = []
            for rr in ru_sorted.itertuples(index=False):
                u_items.append(
                    RegionItem(
                        id=str(rr.entity_id),
                        name=str(rr.entity_name),
                        joint_rank=float(rr.rnk),
                        ranks={u: int(rr.rnk)},
                        counts={u: int(rr.play_count)},
                    )
                )
            regions[f"{u}_only"] = [ri.__dict__ for ri in u_items]
    else:
        u1, u2, u3 = users
        # 3-way intersection
        inter3 = _region_entities((u1, u2, u3), ())
        totals[f"{u1}_{u2}_{u3}"] = int(inter3.shape[0])
        items3 = []
        for row in inter3.head(limit_per_region).itertuples(index=False):
            ranks_counts: dict[str, tuple[int, int]] = {}
            for u in (u1, u2, u3):
                r = per_user[u]
                rr = r[r.entity_id == row.entity_id].sort_values(["rnk", "entity_name"]).iloc[0]
                ranks_counts[u] = (int(rr.rnk), int(rr.play_count))
            items3.append(
                RegionItem(
                    id=str(row.entity_id),
                    name=str(row.entity_name),
                    joint_rank=float(row.joint_rank),
                    ranks={k: v[0] for k, v in ranks_counts.items()},
                    counts={k: v[1] for k, v in ranks_counts.items()},
                ).__dict__
            )
        regions[f"{u1}_{u2}_{u3}"] = items3

        # Exactly-two intersections
        pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            ((u1, u2), (u3,)),
            ((u1, u3), (u2,)),
            ((u2, u3), (u1,)),
        ]
        for req, exc in pairs:
            dfp = _region_entities(req, exc)
            totals["_".join(req)] = int(dfp.shape[0])
            items_pair = []
            for row in dfp.head(limit_per_region).itertuples(index=False):
                ranks_counts: dict[str, tuple[int, int]] = {}
                for u in req:
                    r = per_user[u]
                    rr = r[r.entity_id == row.entity_id].sort_values(["rnk", "entity_name"]).iloc[0]
                    ranks_counts[u] = (int(rr.rnk), int(rr.play_count))
                items_pair.append(
                    RegionItem(
                        id=str(row.entity_id),
                        name=str(row.entity_name),
                        joint_rank=float(row.joint_rank),
                        ranks={k: v[0] for k, v in ranks_counts.items()},
                        counts={k: v[1] for k, v in ranks_counts.items()},
                    ).__dict__
                )
            regions["_".join(req)] = items_pair

        # Non-intersections per user (top overall)
        for u in (u1, u2, u3):
            ru = per_user[u]
            totals[f"{u}_only"] = int(ru["entity_id"].nunique())
            ru_sorted = ru.sort_values(["rnk", "entity_name"]).head(limit_per_region)
            u_items: list[RegionItem] = []
            for rr in ru_sorted.itertuples(index=False):
                u_items.append(
                    RegionItem(
                        id=str(rr.entity_id),
                        name=str(rr.entity_name),
                        joint_rank=float(rr.rnk),
                        ranks={u: int(rr.rnk)},
                        counts={u: int(rr.play_count)},
                    )
                )
            regions[f"{u}_only"] = [ri.__dict__ for ri in u_items]

    return {
        "users": users,
        "mode": mode,
        "regions": regions,
        "totals": totals,
        "user_counts": user_counts,
        "user_labels": user_labels,
    }

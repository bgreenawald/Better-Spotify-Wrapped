"""In-process caches for the dashboard's expensive, DuckDB-backed queries.

Shared by the Wrapped and Trends callbacks, which is why these live apart from
either module.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import pandas as pd


class SimpleLRUCache:
    """Very small, in-process LRU cache for DataFrames keyed by strings.

    Guarded by a lock: Dash under gunicorn or waitress serves callbacks from a
    thread pool, so two requests can interleave a get and a set and corrupt the
    OrderedDict. The lock is far cheaper than the queries it protects.
    """

    def __init__(self, maxsize: int = 20):
        self.maxsize = maxsize
        self._store: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> pd.DataFrame | None:
        with self._lock:
            if key in self._store:
                val = self._store.pop(key)
                self._store[key] = val
                return val
            return None

    def set(self, key: str, value: pd.DataFrame) -> None:
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            self._store[key] = value
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)


# Lightweight in-process LRU caches for expensive, DuckDB-backed computations
GENRE_TRENDS_CACHE = SimpleLRUCache(maxsize=50)
OVERALL_GENRES_CACHE = SimpleLRUCache(maxsize=50)
TOP_ALBUMS_CACHE = SimpleLRUCache(maxsize=50)
TOP_GENRES_WRAPPED_CACHE = SimpleLRUCache(maxsize=50)


def make_filter_cache_key(
    user_id,
    start_date,
    end_date,
    exclude_december,
    remove_incognito,
    excluded_tracks,
    excluded_genres,
    excluded_artists,
    excluded_albums,
) -> str:
    """Create a stable cache key from filter inputs."""

    def _to_tuple(x):
        if x is None:
            return ()
        if isinstance(x, list | tuple | set):
            return tuple(sorted(map(str, x)))
        return (str(x),)

    parts = (
        _to_tuple(user_id),
        _to_tuple(start_date),
        _to_tuple(end_date),
        _to_tuple(bool(exclude_december)),
        _to_tuple(bool(remove_incognito)),
        _to_tuple(excluded_tracks),
        _to_tuple(excluded_genres),
        _to_tuple(excluded_artists),
        _to_tuple(excluded_albums),
    )
    # Hash a repr of the whole tuple rather than joining on "|" and ";": a
    # track actually named "a;b" could otherwise collide with two tracks named
    # "a" and "b", silently serving one filter's results for another.
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def make_wrapped_cache_key(
    user_id,
    selected_year,
    exclude_december,
    remove_incognito,
    excluded_tracks,
    excluded_artists,
    excluded_albums,
    excluded_genres,
) -> str:
    parts = (
        str(user_id or ""),
        selected_year,
        bool(exclude_december),
        bool(remove_incognito),
        tuple(sorted(map(str, excluded_tracks or []))),
        tuple(sorted(map(str, excluded_artists or []))),
        tuple(sorted(map(str, excluded_albums or []))),
        tuple(sorted(map(str, excluded_genres or []))),
    )
    # Hashed for the same reason as make_filter_cache_key: a name containing
    # the separator could otherwise collide with a different filter set.
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

"""Tests for the Spotify API cache layer: null handling, atomicity, recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from spotipy.exceptions import SpotifyException

from src.api.api import SpotifyDataCollector


@pytest.fixture
def collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SpotifyDataCollector:
    """A collector with a temp cache dir and no real Spotify client."""
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test-id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test-secret")
    monkeypatch.delenv("DATA_DIR", raising=False)
    c = SpotifyDataCollector(cache_dir=tmp_path / "cache", rate_limit_delay=0.0)
    # No test here should reach the network.
    c.spotify_client = None  # type: ignore[assignment]
    return c


def test_constructor_does_not_require_data_dir(collector: SpotifyDataCollector) -> None:
    """DB-only commands must not fail on an unset DATA_DIR."""
    assert collector.data_dir is None


def test_null_item_does_not_discard_good_items(collector: SpotifyDataCollector) -> None:
    """A null in the response must not cost the batch its resolved items."""
    requested = ["good1", "dead1", "good2"]

    def api_fn(_batch: list[str]) -> list[Any]:
        return [{"id": "good1", "name": "A"}, None, {"id": "good2", "name": "B"}]

    items = collector._fetch_cached("tracks", requested, 50, api_fn)

    assert sorted(i["id"] for i in items) == ["good1", "good2"]
    assert (collector.cache_dir / "tracks" / "good1.json").exists()
    assert (collector.cache_dir / "tracks" / "good2.json").exists()


def test_unresolvable_id_is_negative_cached(collector: SpotifyDataCollector) -> None:
    """A null ID is recorded so later runs stop re-requesting it."""
    calls: list[list[str]] = []

    def api_fn(batch: list[str]) -> list[Any]:
        calls.append(list(batch))
        return [{"id": "good1", "name": "A"}, None]

    collector._fetch_cached("tracks", ["good1", "dead1"], 50, api_fn)
    assert "dead1" in collector._get_dead_ids("tracks")

    # Second run: good1 is cached, dead1 is known-dead, so no request at all.
    collector._fetch_cached("tracks", ["good1", "dead1"], 50, api_fn)
    assert len(calls) == 1


def test_corrupt_cache_file_is_treated_as_a_miss(collector: SpotifyDataCollector) -> None:
    """Truncated JSON must self-heal by refetching, not crash forever."""
    path = collector.cache_dir / "tracks" / "t1.json"
    path.write_text('{"id": "t1", "na', encoding="utf-8")

    def api_fn(_batch: list[str]) -> list[Any]:
        return [{"id": "t1", "name": "recovered"}]

    items = collector._fetch_cached("tracks", ["t1"], 50, api_fn)

    assert items == [{"id": "t1", "name": "recovered"}]
    assert json.loads(path.read_text())["name"] == "recovered"


def test_save_cache_leaves_no_partial_files(collector: SpotifyDataCollector) -> None:
    """Writes are atomic: no temp files survive a completed save."""
    collector._save_cache("tracks", [{"id": "t1", "name": "A"}])
    leftovers = list((collector.cache_dir / "tracks").glob(".*tmp"))
    assert leftovers == []
    assert json.loads((collector.cache_dir / "tracks" / "t1.json").read_text())["id"] == "t1"


def test_save_cache_skips_nulls_and_idless_items(collector: SpotifyDataCollector) -> None:
    """Null and malformed entries are skipped rather than raising."""
    collector._save_cache("artists", [None, {"no_id": 1}, {"id": "a1"}])  # type: ignore[list-item]
    assert collector._get_cached_ids("artists") == {"a1"}


def test_batch_is_bisected_on_400(collector: SpotifyDataCollector) -> None:
    """One malformed ID must not cost the whole batch."""
    attempted: list[list[str]] = []

    def api_fn(batch: list[str]) -> list[Any]:
        attempted.append(list(batch))
        if "bad" in batch and len(batch) > 1:
            raise SpotifyException(400, -1, "invalid id", headers={})
        if batch == ["bad"]:
            raise SpotifyException(400, -1, "invalid id", headers={})
        return [{"id": i} for i in batch]

    items = collector._fetch_cached("tracks", ["g1", "g2", "bad", "g3"], 50, api_fn)

    # The good IDs survive even though one ID poisons any batch containing it.
    assert sorted(i["id"] for i in items) == ["g1", "g2", "g3"]
    assert len(attempted) > 1, "expected the batch to be bisected"


def test_retry_after_is_bounded(collector: SpotifyDataCollector) -> None:
    """An absurd or missing Retry-After must not stall the run."""
    huge = SpotifyException(429, -1, "slow down", headers={"Retry-After": "99999"})
    assert collector._parse_retry_after(huge, 0) == collector._MAX_RETRY_AFTER

    missing = SpotifyException(429, -1, "slow down", headers={})
    assert collector._parse_retry_after(missing, 2) == 4.0

    junk = SpotifyException(429, -1, "slow down", headers={"Retry-After": "soon"})
    assert collector._parse_retry_after(junk, 1) == 2.0


def test_successful_request_decays_adaptive_delay(collector: SpotifyDataCollector) -> None:
    """The delay walks back to baseline instead of staying throttled."""
    collector._rate_limit_delay = 5.0
    collector._baseline_rate_limit_delay = 1.0

    for _ in range(100):
        collector._handle_successful_request()

    assert collector._rate_limit_delay == pytest.approx(1.0)


def test_total_batch_failure_raises_instead_of_returning_empty(
    collector: SpotifyDataCollector,
) -> None:
    """A wholly failed run must not look like 'nothing to do'.

    Returning no items let the CLI print "Updated rows: 0" and exit 0, so a
    wrong secret was indistinguishable from an already-complete database.
    """

    def api_fn(_batch: list[str]) -> list[Any]:
        raise SpotifyException(401, -1, "invalid client", headers={})

    with pytest.raises(RuntimeError, match="SPOTIFY_CLIENT_ID"):
        collector._fetch_cached("tracks", ["t1", "t2"], 50, api_fn)


def test_partial_batch_failure_still_returns_what_succeeded(
    collector: SpotifyDataCollector,
) -> None:
    """One bad batch must not discard the batches that worked."""
    calls: list[list[str]] = []

    def api_fn(batch: list[str]) -> list[Any]:
        calls.append(list(batch))
        if "boom" in batch:
            raise SpotifyException(500, -1, "server error", headers={})
        return [{"id": i} for i in batch]

    # batch_size 1 gives one batch per id, so exactly one of them fails.
    items = collector._fetch_cached("tracks", ["good1", "boom", "good2"], 1, api_fn)

    assert sorted(i["id"] for i in items) == ["good1", "good2"]

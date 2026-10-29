"""Tests for the distribution measures and the monthly diversity metric."""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics.albums import gini_coefficient, normalized_entropy, shannon_entropy
from src.metrics.trends import get_listening_diversity_by_month

# --------------------------------------------------------------------------
# Pure measures
# --------------------------------------------------------------------------


def test_even_distribution_is_maximally_diverse() -> None:
    """Eight artists with equal plays: entropy is log2(8) and Gini is 0."""
    counts = [10] * 8
    assert shannon_entropy(counts) == pytest.approx(3.0)
    assert normalized_entropy(counts) == pytest.approx(1.0)
    assert gini_coefficient(counts) == pytest.approx(0.0, abs=1e-9)


def test_concentrated_distribution_is_not_diverse() -> None:
    """One artist taking nearly everything: low entropy, high Gini."""
    counts = [100, 1, 1, 1]
    assert shannon_entropy(counts) < 0.5
    assert normalized_entropy(counts) < 0.2
    assert gini_coefficient(counts) > 0.7


def test_single_entity_has_no_diversity() -> None:
    """One artist cannot be spread across anything."""
    assert shannon_entropy([5]) == 0.0
    assert normalized_entropy([5]) == 0.0
    assert gini_coefficient([5]) == 0.0


def test_empty_and_nonpositive_inputs_are_safe() -> None:
    """No plays must not raise or divide by zero."""
    for fn in (shannon_entropy, normalized_entropy, gini_coefficient):
        assert fn([]) == 0.0
        assert fn([0, 0]) == 0.0


def test_normalized_entropy_is_comparable_across_sizes() -> None:
    """Raw entropy grows with n; the normalized form does not.

    This is why the normalized value is the one worth plotting month to month.
    """
    assert shannon_entropy([1] * 4) < shannon_entropy([1] * 16)
    assert normalized_entropy([1] * 4) == pytest.approx(normalized_entropy([1] * 16))


def test_entropy_ordering_matches_intuition() -> None:
    """Spreading plays out raises entropy and lowers concentration."""
    concentrated = [50, 1, 1]
    spread = [17, 17, 18]
    assert shannon_entropy(spread) > shannon_entropy(concentrated)
    assert gini_coefficient(spread) < gini_coefficient(concentrated)


# --------------------------------------------------------------------------
# Monthly metric
# --------------------------------------------------------------------------


def _plays(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Build a plays frame from (month, artist, count) triples."""
    out: list[dict] = []
    for month, artist, n in rows:
        for _ in range(n):
            out.append(
                {
                    "ts": pd.Timestamp(f"{month}-15 12:00:00"),
                    "master_metadata_album_artist_name": artist,
                    "artist_id": f"id_{artist}",
                }
            )
    return pd.DataFrame(out)


def test_empty_input_returns_the_declared_columns() -> None:
    """An empty frame yields an empty result with a stable schema."""
    out = get_listening_diversity_by_month(pd.DataFrame())
    for col in ("month", "entropy", "entropy_norm", "gini", "top_artist_share"):
        assert col in out.columns


def test_a_varied_month_scores_higher_than_a_repetitive_one() -> None:
    """The metric's whole purpose, on two contrived months."""
    plays = _plays(
        [("2024-01", "A", 10), ("2024-01", "B", 10), ("2024-01", "C", 10), ("2024-01", "D", 10)]
        + [("2024-02", "A", 28), ("2024-02", "B", 1), ("2024-02", "C", 1)]
    )

    out = get_listening_diversity_by_month(plays).set_index("month")

    assert out.loc["2024-01", "entropy_norm"] > out.loc["2024-02", "entropy_norm"]
    assert out.loc["2024-01", "gini"] < out.loc["2024-02", "gini"]
    assert out.loc["2024-02", "top_artist_share"] > 90


def test_counts_and_shares_are_right() -> None:
    """Plays, unique artists and the top share are straightforward counts."""
    plays = _plays([("2024-03", "A", 6), ("2024-03", "B", 2), ("2024-03", "C", 2)])

    row = get_listening_diversity_by_month(plays).iloc[0]

    assert int(row["plays"]) == 10
    assert int(row["unique_artists"]) == 3
    assert row["top_artist_share"] == pytest.approx(60.0)


def test_artists_are_keyed_by_id_not_name() -> None:
    """A renamed artist is one entity, matching the other metrics."""
    plays = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2024-04-10"),
                "master_metadata_album_artist_name": "Old Name",
                "artist_id": "ar1",
            },
            {
                "ts": pd.Timestamp("2024-04-11"),
                "master_metadata_album_artist_name": "New Name",
                "artist_id": "ar1",
            },
        ]
    )

    row = get_listening_diversity_by_month(plays).iloc[0]

    assert int(row["unique_artists"]) == 1
    assert row["entropy"] == 0.0


def test_missing_artist_id_falls_back_to_the_name() -> None:
    """Rows without an id are still counted, by name."""
    plays = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-05-01"), "master_metadata_album_artist_name": "A"},
            {"ts": pd.Timestamp("2024-05-02"), "master_metadata_album_artist_name": "B"},
        ]
    )

    row = get_listening_diversity_by_month(plays).iloc[0]
    assert int(row["unique_artists"]) == 2

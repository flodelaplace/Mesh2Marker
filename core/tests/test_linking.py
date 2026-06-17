"""Marker <-> vertex linking: centroid, link set, records, validation."""

import numpy as np
import pytest

from mesh2marker.linking import (
    LinkSet,
    MarkerLink,
    centroid_vertex,
    ordered_indices,
    validate_against_known,
)

# verts 1, 3, 4 are positioned so 4 is exactly at the centroid of {1, 3, 4}.
VERTS = np.array(
    [
        [0.0, 0.0, 0.0],  # 0 (unused)
        [0.0, 0.0, 0.0],  # 1
        [0.0, 0.0, 0.0],  # 2 (unused)
        [10.0, 0.0, 0.0],  # 3
        [5.0, 0.0, 0.0],  # 4 (centroid of 1, 3, 4)
    ]
)


def test_centroid_single_index_returns_itself():
    assert centroid_vertex(VERTS, [3]) == 3


def test_centroid_multiple_returns_nearest_real_index():
    # centroid of (0,0,0),(10,0,0),(5,0,0) is (5,0,0) -> vertex 4.
    assert centroid_vertex(VERTS, [1, 3, 4]) == 4


def test_centroid_empty_raises():
    with pytest.raises(ValueError):
        centroid_vertex(VERTS, [])


def test_ordered_indices_places_centroid_first():
    assert ordered_indices(VERTS, [1, 3, 4]) == [4, 1, 3]
    assert ordered_indices(VERTS, [3]) == [3]


def test_ordered_indices_empty_raises():
    with pytest.raises(ValueError):
        ordered_indices(VERTS, [])


def test_add_get_remove():
    ls = LinkSet()
    ls.add_link("RASI", [100])
    assert "RASI" in ls
    assert ls.get("RASI").vertex_indices == [100]
    assert len(ls) == 1
    ls.remove_link("RASI")
    assert ls.get("RASI") is None
    assert len(ls) == 0


def test_add_link_centroid_first_with_verts():
    ls = LinkSet()
    link = ls.add_link("M", [1, 3, 4], verts=VERTS)
    assert link.vertex_indices[0] == 4  # centroid first
    assert sorted(link.vertex_indices) == [1, 3, 4]
    assert link.chosen_index == 4


def test_add_link_empty_raises():
    with pytest.raises(ValueError):
        LinkSet().add_link("M", [])


def test_add_link_replaces_existing():
    ls = LinkSet()
    ls.add_link("M", [1])
    ls.add_link("M", [2, 3])
    assert ls.get("M").vertex_indices == [2, 3]
    assert len(ls) == 1


def test_records_roundtrip():
    ls = LinkSet()
    ls.add_link("RASI", [100])
    ls.add_link("RKNE", [4, 1, 3])
    records = ls.to_records()
    assert records == [
        {"marker": "RASI", "vertex_indices": [100]},
        {"marker": "RKNE", "vertex_indices": [4, 1, 3]},
    ]
    restored = LinkSet.from_records(records)
    assert restored.to_records() == records
    assert restored.get("RKNE").chosen_index == 4


def test_marker_link_chosen_index_empty_raises():
    with pytest.raises(ValueError):
        _ = MarkerLink("M", []).chosen_index


def test_validate_against_known_match_and_mismatch():
    ls = LinkSet()
    ls.add_link("RASI", [100])  # matches
    ls.add_link("RKNE", [201])  # mismatches expected 200
    ls.add_link("LASI", [1, 3, 4], verts=VERTS)  # centroid 4 matches
    known = {"RASI": 100, "RKNE": 200, "LASI": 4, "MISSING": 7}

    result = validate_against_known(ls, known)

    assert result == {"RASI": True, "RKNE": False, "LASI": True}
    assert "MISSING" not in result  # not in linkset

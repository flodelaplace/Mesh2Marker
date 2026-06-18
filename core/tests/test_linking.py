"""Marker <-> vertex linking: centroid, link set, records, validation."""

import numpy as np
import pytest

from mesh2marker.kinematics import euler_xyz_to_matrix
from mesh2marker.linking import (
    LinkSet,
    MarkerLink,
    auto_link_markers,
    centroid_vertex,
    marker_local_from_vertex,
    nearest_vertex,
    ordered_indices,
    reposition_marker_to_vertex,
    suggest_fixed,
    suggest_fixed_map,
    validate_against_known,
)
from mesh2marker.markers import marker_world_positions
from mesh2marker.osim import OsimBody, OsimFrameOffset, OsimJoint, OsimMarker, OsimModel
from mesh2marker.procrustes import SimilarityTransform

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


def test_nearest_vertex_known_case():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    assert nearest_vertex(verts, [0.9, 0.0, 0.0]) == 1
    assert nearest_vertex(verts, [0.0, 0.0, 0.1]) == 0


def test_nearest_vertex_empty_raises():
    with pytest.raises(ValueError):
        nearest_vertex(np.zeros((0, 3)), [0.0, 0.0, 0.0])


def _identity_similarity() -> SimilarityTransform:
    return SimilarityTransform(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )


def _one_body_model(markers: list[OsimMarker]) -> OsimModel:
    # body "B" hangs off ground with identity offset -> world[B] is identity, so a
    # marker's neutral world position equals its location.
    body = OsimBody("B", [])
    joint = OsimJoint(
        name="ground_B",
        joint_type="CustomJoint",
        parent_body="ground",
        child_body="B",
        parent_offset=OsimFrameOffset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        child_offset=OsimFrameOffset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        coordinates=[],
    )
    return OsimModel(name="m", bodies=[body], joints=[joint], markers=markers)


def test_auto_link_markers_picks_nearest():
    markers = [
        OsimMarker("m1", "B", [0.0, 0.0, 0.0]),
        OsimMarker("m2", "B", [10.0, 0.0, 0.0]),
    ]
    model = _one_body_model(markers)
    verts = np.array(
        [
            [0.1, 0.0, 0.0],  # 0 -> nearest to m1
            [9.9, 0.0, 0.0],  # 1 -> nearest to m2
            [100.0, 100.0, 100.0],  # 2 -> far from both
        ]
    )
    # Identity global + no seg: verts and markers are in the same (neutral) frame.
    result = auto_link_markers(
        model, verts, _identity_similarity(), seg_transforms=None
    )
    assert result == {"m1": 0, "m2": 1}


def test_auto_link_markers_identity_seg_matches_neutral():
    markers = [
        OsimMarker("m1", "B", [0.0, 0.0, 0.0]),
        OsimMarker("m2", "B", [10.0, 0.0, 0.0]),
    ]
    model = _one_body_model(markers)
    verts = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    seg = {"B": np.eye(4)}
    result = auto_link_markers(model, verts, _identity_similarity(), seg_transforms=seg)
    assert result == {"m1": 0, "m2": 1}


def _body_model(markers, offset_trans=(0.0, 0.0, 0.0), offset_ori=(0.0, 0.0, 0.0)):
    body = OsimBody("B", [])
    joint = OsimJoint(
        name="ground_B",
        joint_type="CustomJoint",
        parent_body="ground",
        child_body="B",
        parent_offset=OsimFrameOffset(list(offset_trans), list(offset_ori)),
        child_offset=OsimFrameOffset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        coordinates=[],
    )
    return OsimModel(name="m", bodies=[body], joints=[joint], markers=markers)


def test_marker_local_from_vertex_roundtrip():
    # Non-trivial body world (offset translation + orientation) and a non-trivial
    # per-segment correction (scale + rotation + translation).
    local = np.array([0.03, -0.05, 0.02])
    model = _body_model(
        [OsimMarker("M0", "B", list(local))],
        offset_trans=(0.1, 0.2, 0.3),
        offset_ori=(0.3, -0.2, 0.5),
    )
    seg_rot = euler_xyz_to_matrix([0.4, 0.1, -0.3])
    seg = np.eye(4)
    seg[:3, :3] = 1.3 * seg_rot
    seg[:3, 3] = [0.5, -0.1, 0.2]
    seg_transforms = {"B": seg}

    # Forward chain (markers module) then inverse (linking) must return local.
    world = marker_world_positions(model, seg_transforms=seg_transforms)["M0"]
    back = marker_local_from_vertex(model, "M0", world, seg_transforms)
    np.testing.assert_allclose(back, local, atol=1e-9)


def test_reposition_marker_identity_returns_vertex():
    model = _body_model([OsimMarker("M0", "B", [0.0, 0.0, 0.0])])  # world[B] = identity
    verts = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    local = reposition_marker_to_vertex(
        model, "M0", verts, 1, _identity_similarity(), seg_transforms=None
    )
    np.testing.assert_allclose(local, [1.0, 2.0, 3.0], atol=1e-9)


def test_reposition_out_of_range_raises():
    model = _body_model([OsimMarker("M0", "B", [0.0, 0.0, 0.0])])
    verts = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="out of range"):
        reposition_marker_to_vertex(model, "M0", verts, 5, _identity_similarity())


def test_reposition_unknown_marker_raises():
    model = _body_model([OsimMarker("M0", "B", [0.0, 0.0, 0.0])])
    verts = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown marker"):
        reposition_marker_to_vertex(model, "NOPE", verts, 0, _identity_similarity())


def test_validate_against_known_match_and_mismatch():
    ls = LinkSet()
    ls.add_link("RASI", [100])  # matches
    ls.add_link("RKNE", [201])  # mismatches expected 200
    ls.add_link("LASI", [1, 3, 4], verts=VERTS)  # centroid 4 matches
    known = {"RASI": 100, "RKNE": 200, "LASI": 4, "MISSING": 7}

    result = validate_against_known(ls, known)

    assert result == {"RASI": True, "RKNE": False, "LASI": True}
    assert "MISSING" not in result  # not in linkset


def test_suggest_fixed_bony_cases():
    for name in ("RASI", "RLMAL", "RMFC", "RACR", "C7", "HTOP", "Nose", "RWrist_hand"):
        assert suggest_fixed(name) is True, name


def test_suggest_fixed_soft_and_ambiguous_cases():
    soft = ("RFLT", "RFLB", "RSHN", "RTIB", "RFAradius", "RHTO", "RFRM", "c_spine0")
    for name in soft:
        assert suggest_fixed(name) is False, name


def test_suggest_fixed_case_insensitive():
    assert suggest_fixed("rasi") is True
    assert suggest_fixed("c7") is True
    assert suggest_fixed("rtib") is False


def test_suggest_fixed_map():
    result = suggest_fixed_map(["RASI", "RTIB", "C7"])
    assert result == {"RASI": True, "RTIB": False, "C7": True}

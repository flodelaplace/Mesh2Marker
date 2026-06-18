"""Session-links <-> correspondence-file bridge, reload, and export validation."""

import importlib.util

import pytest

from mesh2marker.correspondence import (
    _to_contract_map,
    correspondence_to_links,
    links_to_correspondence,
    read_correspondence_links,
    validate_for_export,
    write_correspondence,
)

_HAS_CONTRACTS = importlib.util.find_spec("mesh2sim.contracts") is not None

META = {
    "mhr_topology_id": "mhr-template-v1",
    "opensim_model": "Pose2Sim_Wholebody",
    "marker_set": "pose2sim_wholebody_73",
}


def _links() -> list[dict]:
    return [
        {
            "marker": "RASI",
            "vertex_indices": [10, 11],
            "opensim_body": "pelvis",
            "local_offset": [0.1, -0.2, 0.3],
            "fixed": True,
            "synthpose_index": 0,
        },
        {
            "marker": "RLFC",
            "vertex_indices": [20],
            "opensim_body": "femur_r",
            "local_offset": [0.0, 0.0, 0.0],
            "fixed": False,
            "synthpose_index": None,
        },
    ]


def test_bridge_builds_schema():
    corr = links_to_correspondence(_links(), **META)
    assert corr.schema_version == "0.1.0"
    assert corr.mhr_topology_id == "mhr-template-v1"
    assert corr.opensim_model == "Pose2Sim_Wholebody"
    assert [m.name for m in corr.markers] == ["RASI", "RLFC"]
    rasi = corr.markers[0]
    assert rasi.mhr_vertices == [10, 11]
    assert rasi.opensim_body == "pelvis"
    assert rasi.local_offset == [0.1, -0.2, 0.3]
    assert rasi.fixed is True
    assert rasi.synthpose_index == 0


def test_links_to_links_roundtrip_in_memory():
    links = _links()
    corr = links_to_correspondence(links, **META)
    assert correspondence_to_links(corr) == links


def test_write_read_roundtrip(tmp_path):
    links = _links()
    path = tmp_path / "correspondence.json"
    write_correspondence(links, path, **META)

    corr, reloaded = read_correspondence_links(path)
    assert corr.mhr_topology_id == "mhr-template-v1"
    assert corr.opensim_model == "Pose2Sim_Wholebody"
    assert reloaded == links


def test_default_frame_alignment_is_identity():
    corr = links_to_correspondence(_links(), **META)
    assert corr.frame_alignment.rotation == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert corr.frame_alignment.translation == [0.0, 0.0, 0.0]
    assert corr.frame_alignment.scale == 1.0


def test_default_identity_values():
    # No identity kwargs -> canonical contract values.
    corr = links_to_correspondence(_links())
    assert corr.opensim_model == "Pose2Sim_Wholebody"
    assert corr.mhr_topology_id == "mhr_v1"
    assert corr.marker_set == "mesh2marker"
    assert corr.schema_version == "0.1.0"


def test_export_rejects_unknown_marker(tmp_path):
    links = [
        {"marker": "NOT_A_MARKER", "vertex_indices": [1], "opensim_body": "pelvis"}
    ]
    with pytest.raises(ValueError, match="unknown marker name"):
        write_correspondence(links, tmp_path / "x.json")


def test_export_rejects_unknown_body(tmp_path):
    links = [{"marker": "RASI", "vertex_indices": [1], "opensim_body": "NOT_A_BODY"}]
    with pytest.raises(ValueError, match="unknown opensim body"):
        write_correspondence(links, tmp_path / "x.json")


def test_valid_export_passes_validation(tmp_path):
    # The RASI / RWrist_hand example with canonical identity validates and writes.
    links = [
        {
            "marker": "RASI",
            "vertex_indices": [7696],
            "opensim_body": "pelvis",
            "local_offset": [0.0123, 0.0181, 0.1285],
            "fixed": True,
            "synthpose_index": 11,
        },
        {
            "marker": "RWrist_hand",
            "vertex_indices": [41, 402, 517],
            "opensim_body": "hand_r",
            "local_offset": [0.0, 0.0, 0.0],
            "fixed": False,
            "synthpose_index": None,
        },
    ]
    corr = write_correspondence(links, tmp_path / "ok.json")
    assert (tmp_path / "ok.json").is_file()
    assert corr.opensim_model == "Pose2Sim_Wholebody"
    validate_for_export(corr)  # no raise


@pytest.mark.skipif(not _HAS_CONTRACTS, reason="mesh2sim.contracts not installed")
def test_builds_a_valid_contract_map():
    corr = links_to_correspondence(_links())
    contract_map = _to_contract_map(corr)
    assert contract_map.opensim_model == "Pose2Sim_Wholebody"
    assert contract_map.mhr_topology_id == "mhr_v1"
    assert len(contract_map.markers) == len(_links())

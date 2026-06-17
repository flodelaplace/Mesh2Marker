"""Session-links <-> correspondence-file bridge and reload."""

from mesh2marker.correspondence import (
    correspondence_to_links,
    links_to_correspondence,
    read_correspondence_links,
    write_correspondence,
)

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
    assert corr.schema_version == "1.0"
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

"""Parse the committed minimal fixture; optionally parse the real local model."""

from pathlib import Path

import pytest

from mesh2marker.osim import parse_osim

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.osim"
REAL_MODEL = (
    Path(__file__).parents[2] / "local_models" / "Model_Flodelaplace_mocap.osim"
)


@pytest.fixture
def model():
    return parse_osim(FIXTURE)


def test_model_name(model):
    assert model.name == "minimal"


def test_bodies(model):
    assert [b.name for b in model.bodies] == ["pelvis", "femur_r", "tibia_r"]


def test_geometries(model):
    by_name = {b.name: b for b in model.bodies}

    # FrameGeometry ignored: pelvis exposes exactly its one attached mesh.
    assert [g.mesh_file for g in by_name["pelvis"].geometries] == ["sacrum.vtp"]

    # A body can carry several meshes.
    femur = by_name["femur_r"]
    assert len(femur.geometries) == 2
    assert [g.mesh_file for g in femur.geometries] == [
        "femur_r.vtp",
        "femur_cap_r.vtp",
    ]
    assert femur.geometries[1].scale_factors == [1.1, 1.0, 0.9]

    assert [g.mesh_file for g in by_name["tibia_r"].geometries] == ["tibia_r.vtp"]


def test_joints_resolved(model):
    by_name = {j.name: j for j in model.joints}
    assert set(by_name) == {"ground_pelvis", "hip_r"}

    gp = by_name["ground_pelvis"]
    assert gp.joint_type == "CustomJoint"
    assert gp.parent_body == "ground"  # resolved via /ground
    assert gp.child_body == "pelvis"
    assert gp.coordinates == ["pelvis_tilt", "pelvis_tx", "pelvis_ty"]

    hip = by_name["hip_r"]
    assert hip.parent_body == "pelvis"  # resolved via path-form socket
    assert hip.child_body == "femur_r"
    assert hip.coordinates == ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"]
    assert hip.parent_offset.translation == [-0.07, -0.06, 0.08]


def test_adjacency(model):
    adj = model.adjacency()
    assert adj["ground"] == ["pelvis"]
    assert adj["pelvis"] == ["femur_r"]


def test_markers_exactly_two_despite_noise(model):
    # The ForceSet muscle path points and the wrap object also carry
    # socket_parent_frame tags; strict scoping must ignore them.
    assert len(model.markers) == 2

    by_name = {m.name: m for m in model.markers}
    assert set(by_name) == {"RASIS", "RKNE"}
    assert by_name["RASIS"].parent_body == "pelvis"
    assert by_name["RASIS"].location == [0.01, -0.02, 0.03]
    assert by_name["RKNE"].parent_body == "femur_r"
    assert by_name["RKNE"].location == [0.0, -0.4, 0.05]


@pytest.mark.skipif(
    not REAL_MODEL.exists(), reason="real local model not present (CI / clean checkout)"
)
def test_real_model():
    model = parse_osim(REAL_MODEL)
    assert len(model.bodies) == 30
    assert len(model.markers) == 73

    body_names = {b.name for b in model.bodies}
    for m in model.markers:
        assert m.parent_body in body_names, (
            f"marker {m.name!r} -> unknown body {m.parent_body!r}"
        )

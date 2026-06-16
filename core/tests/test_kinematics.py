"""Forward kinematics: synthetic exactness + real-model symmetry."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.kinematics import (
    Transform,
    euler_xyz_to_matrix,
    forward_kinematics,
    joint_centers,
)
from mesh2marker.osim import (
    OsimBody,
    OsimFrameOffset,
    OsimJoint,
    OsimModel,
    parse_osim,
)

REAL_MODEL = (
    Path(__file__).parents[2] / "local_models" / "Model_Flodelaplace_mocap.osim"
)


def test_euler_z_90():
    rot = euler_xyz_to_matrix([0.0, 0.0, np.pi / 2])
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rot, expected, atol=1e-9)


def test_euler_xyz_order():
    a, b, c = 0.3, -0.5, 0.8
    rx = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    ry = np.array(
        [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
    )
    rz = np.array(
        [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(euler_xyz_to_matrix([a, b, c]), rx @ ry @ rz, atol=1e-9)


def _offset(translation, orientation=(0.0, 0.0, 0.0)):
    return OsimFrameOffset(translation=list(translation), orientation=list(orientation))


def _chain_model() -> OsimModel:
    """ground -> A -> B. A is a pure translation; B adds a known Z rotation."""
    return OsimModel(
        name="chain",
        bodies=[OsimBody("A", []), OsimBody("B", [])],
        joints=[
            OsimJoint(
                name="ground_A",
                joint_type="CustomJoint",
                parent_body="ground",
                child_body="A",
                parent_offset=_offset([1.0, 0.0, 0.0]),
                child_offset=_offset([0.0, 0.0, 0.0]),
                coordinates=[],
            ),
            OsimJoint(
                name="A_B",
                joint_type="CustomJoint",
                parent_body="A",
                child_body="B",
                parent_offset=_offset([0.0, 1.0, 0.0], [0.0, 0.0, np.pi / 2]),
                child_offset=_offset([0.0, 0.0, 0.0]),
                coordinates=[],
            ),
        ],
        markers=[],
    )


def test_chain_world_transforms():
    world = forward_kinematics(_chain_model())

    np.testing.assert_allclose(world["ground"].rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(world["ground"].translation, np.zeros(3), atol=1e-9)

    # A: pure translation along +X, no rotation.
    np.testing.assert_allclose(world["A"].rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(world["A"].translation, [1.0, 0.0, 0.0], atol=1e-9)

    # B = T([1,0,0], I) ∘ T([0,1,0], Rz90): rotation Rz90, translation [1,1,0].
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(world["B"].rotation, rz90, atol=1e-9)
    np.testing.assert_allclose(world["B"].translation, [1.0, 1.0, 0.0], atol=1e-9)


def test_chain_joint_centers():
    centers = joint_centers(_chain_model())
    np.testing.assert_allclose(centers["ground_A"], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(centers["A_B"], [1.0, 1.0, 0.0], atol=1e-9)


def test_compose_and_inverse_roundtrip():
    t = Transform(euler_xyz_to_matrix([0.2, -0.4, 1.1]), np.array([3.0, -1.0, 2.0]))
    roundtrip = t.compose(t.inverse())
    np.testing.assert_allclose(roundtrip.rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(roundtrip.translation, np.zeros(3), atol=1e-9)


def test_missing_parent_raises():
    model = OsimModel(
        name="broken",
        bodies=[OsimBody("orphan", [])],
        joints=[
            OsimJoint(
                name="dangling",
                joint_type="CustomJoint",
                parent_body="nonexistent",  # never reachable from ground
                child_body="orphan",
                parent_offset=_offset([0.0, 0.0, 0.0]),
                child_offset=_offset([0.0, 0.0, 0.0]),
                coordinates=[],
            )
        ],
        markers=[],
    )
    with pytest.raises(ValueError):
        forward_kinematics(model)


@pytest.mark.skipif(
    not REAL_MODEL.exists(), reason="real local model not present (CI / clean checkout)"
)
def test_real_model_structure():
    model = parse_osim(REAL_MODEL)
    world = forward_kinematics(model)

    # Every body is reached from ground.
    assert {b.name for b in model.bodies} <= set(world)

    centers = joint_centers(model)
    tol = 1e-6
    for right, left in [
        ("hip_r", "hip_l"),
        ("walker_knee_r", "walker_knee_l"),
        ("ankle_r", "ankle_l"),
    ]:
        cr = np.array(centers[right])
        cl = np.array(centers[left])
        assert abs(cr[0] - cl[0]) < tol  # same x
        assert abs(cr[1] - cl[1]) < tol  # same y
        assert abs(cr[2] + cl[2]) < tol  # opposite z

    # Vertical order down the right leg.
    assert centers["hip_r"][1] > centers["walker_knee_r"][1] > centers["ankle_r"][1]

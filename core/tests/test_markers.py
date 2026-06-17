"""World positions of OpenSim markers, neutral and under per-segment alignment."""

import numpy as np

from mesh2marker.markers import marker_world_positions
from mesh2marker.osim import OsimBody, OsimFrameOffset, OsimJoint, OsimMarker, OsimModel
from mesh2marker.procrustes import SimilarityTransform


def _model(markers: list[OsimMarker]) -> OsimModel:
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


def _markers() -> list[OsimMarker]:
    return [
        OsimMarker("m1", "B", [1.0, 2.0, 3.0]),
        OsimMarker("m2", "B", [0.0, 0.0, 1.0]),
    ]


def test_neutral_positions():
    positions = marker_world_positions(_model(_markers()))
    np.testing.assert_allclose(positions["m1"], [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(positions["m2"], [0.0, 0.0, 1.0], atol=1e-9)


def test_seg_transform_translation():
    matrix = np.eye(4)
    matrix[:3, 3] = [10.0, 0.0, 0.0]
    positions = marker_world_positions(_model(_markers()), seg_transforms={"B": matrix})
    np.testing.assert_allclose(positions["m1"], [11.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(positions["m2"], [10.0, 0.0, 1.0], atol=1e-9)


def test_seg_transform_rotation_and_scale():
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = 2.0 * rz90
    matrix[:3, 3] = [1.0, 1.0, 1.0]
    positions = marker_world_positions(_model(_markers()), seg_transforms={"B": matrix})
    # m1 = (1,2,3): 2*Rz90@(1,2,3) + (1,1,1) = 2*(-2,1,3)+(1,1,1) = (-3,3,7).
    np.testing.assert_allclose(positions["m1"], [-3.0, 3.0, 7.0], atol=1e-9)


def test_global_transform_used_when_no_seg():
    gt = SimilarityTransform(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[5.0, 0.0, 0.0],
        scale=1.0,
    )
    positions = marker_world_positions(_model(_markers()), global_transform=gt)
    np.testing.assert_allclose(positions["m1"], [6.0, 2.0, 3.0], atol=1e-9)


def test_seg_takes_precedence_over_global():
    matrix = np.eye(4)
    matrix[:3, 3] = [10.0, 0.0, 0.0]
    gt = SimilarityTransform(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[5.0, 0.0, 0.0],
        scale=1.0,
    )
    positions = marker_world_positions(
        _model(_markers()), seg_transforms={"B": matrix}, global_transform=gt
    )
    np.testing.assert_allclose(positions["m1"], [11.0, 2.0, 3.0], atol=1e-9)

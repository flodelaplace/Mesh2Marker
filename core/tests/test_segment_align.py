"""Per-segment skeleton alignment: minimal_rotation, synthetic exact, real model."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.alignment import align_mhr_to_opensim
from mesh2marker.kinematics import joint_centers
from mesh2marker.mhr import MhrSample
from mesh2marker.osim import OsimBody, OsimFrameOffset, OsimJoint, OsimModel
from mesh2marker.procrustes import SimilarityTransform
from mesh2marker.segment_align import (
    INHERIT,
    SEGMENT_TABLE,
    compute_segment_transforms,
    minimal_rotation,
)

REAL_NPZ = Path(__file__).parents[2] / "local_models" / "markers_Squat_mesh.npz"
REAL_OSIM = (
    Path(__file__).parents[2] / "local_models" / "Model_Flodelaplace_mocap.osim"
)


def _apply(matrix: np.ndarray, point) -> np.ndarray:
    return (matrix @ np.array([*point, 1.0]))[:3]


def _identity_similarity() -> SimilarityTransform:
    return SimilarityTransform(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )


def _joint(name: str, parent: str, child: str, translation) -> OsimJoint:
    return OsimJoint(
        name=name,
        joint_type="CustomJoint",
        parent_body=parent,
        child_body=child,
        parent_offset=OsimFrameOffset(
            translation=list(translation), orientation=[0, 0, 0]
        ),
        child_offset=OsimFrameOffset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        coordinates=[],
    )


# --- minimal_rotation ------------------------------------------------------


def test_minimal_rotation_90deg():
    rot = minimal_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rot @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)


def test_minimal_rotation_parallel_is_identity():
    rot = minimal_rotation([0.0, 0.0, 2.0], [0.0, 0.0, 5.0])
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_minimal_rotation_antiparallel():
    a = np.array([1.0, 0.0, 0.0])
    rot = minimal_rotation(a, -a)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(rot @ a, -a, atol=1e-9)


# --- synthetic compute_segment_transforms ----------------------------------


def _sample_with_kps(kps: dict) -> MhrSample:
    keypoints = np.zeros((70, 3), dtype=np.float32)
    for idx, value in kps.items():
        keypoints[idx] = value
    return MhrSample(
        verts=np.zeros((3, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=keypoints,
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )


def _femur_test_model() -> OsimModel:
    # hip_r and walker_knee_r hang off ground -> their centres are the offsets.
    c_hip = [0.1, 0.9, 0.05]
    c_knee = [0.1, 0.5, 0.05]
    names = ("femur_r", "tibia_r", "patella_r", "pelvis", "torso")
    bodies = [OsimBody(n, []) for n in names]
    joints = [
        _joint("hip_r", "ground", "femur_r", c_hip),
        _joint("walker_knee_r", "ground", "tibia_r", c_knee),
        _joint("patellofemoral_r", "femur_r", "patella_r", [0, 0, 0]),
        _joint("ground_pelvis", "ground", "pelvis", [0, 0, 0]),
        _joint("back", "pelvis", "torso", [0, 0, 0]),
    ]
    return OsimModel(name="seg", bodies=bodies, joints=joints, markers=[])


def test_segment_maps_joint_centers_onto_keypoints():
    model = _femur_test_model()
    centers = joint_centers(model)
    kp_p = [1.0, 2.0, 3.0]
    kp_d = [1.0, 1.0, 3.0]
    sample = _sample_with_kps({10: kp_p, 12: kp_d})

    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )

    femur = transforms["femur_r"]
    # global transform is identity, so targets == the raw keypoints.
    np.testing.assert_allclose(_apply(femur, centers["hip_r"]), kp_p, atol=1e-9)
    np.testing.assert_allclose(_apply(femur, centers["walker_knee_r"]), kp_d, atol=1e-9)

    # Trunk receives the identity.
    np.testing.assert_allclose(transforms["pelvis"], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(transforms["torso"], np.eye(4), atol=1e-12)

    # Inheritance: patella copies the femur correction.
    np.testing.assert_array_equal(transforms["patella_r"], femur)


def test_every_body_has_a_correction():
    model = _femur_test_model()
    sample = _sample_with_kps({10: [1.0, 2.0, 3.0], 12: [1.0, 1.0, 3.0]})
    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )
    assert {b.name for b in model.bodies} <= set(transforms)


# --- real model ------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_OSIM.exists()),
    reason="real npz/osim not present (CI / clean checkout)",
)
def test_real_segment_transforms():
    from mesh2marker.mhr import load_mhr_npz
    from mesh2marker.osim import parse_osim

    sample = load_mhr_npz(REAL_NPZ)
    model = parse_osim(REAL_OSIM)
    global_transform, _, _ = align_mhr_to_opensim(sample, model)
    transforms = compute_segment_transforms(sample, model, global_transform)
    centers = joint_centers(model)

    # Each long bone maps its joint centres exactly onto the MHR targets.
    for body, (joint_p, joint_d, kp_p, kp_d) in SEGMENT_TABLE.items():
        matrix = transforms[body]
        t_p = np.asarray(
            global_transform.apply(np.asarray(sample.keypoints[kp_p], dtype=float))
        )
        t_d = np.asarray(
            global_transform.apply(np.asarray(sample.keypoints[kp_d], dtype=float))
        )
        np.testing.assert_allclose(_apply(matrix, centers[joint_p]), t_p, atol=1e-9)
        np.testing.assert_allclose(_apply(matrix, centers[joint_d]), t_d, atol=1e-9)

    # All 30 bodies have a correction.
    assert {b.name for b in model.bodies} <= set(transforms)

    # Inheritance is wired correctly.
    for child, source in INHERIT.items():
        np.testing.assert_array_equal(transforms[child], transforms[source])

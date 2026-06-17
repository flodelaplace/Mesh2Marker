"""Procrustes pre-alignment: synthetic exact recovery, guards, real model."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.alignment import (
    MHR_KP_TO_OSIM_JOINT,
    align_mhr_to_opensim,
    build_alignment_clouds,
    similarity_to_matrix,
)
from mesh2marker.kinematics import euler_xyz_to_matrix
from mesh2marker.mhr import MhrSample
from mesh2marker.osim import OsimBody, OsimFrameOffset, OsimJoint, OsimModel

REAL_NPZ = Path(__file__).parents[2] / "local_models" / "markers_Squat_mesh.npz"
REAL_OSIM = (
    Path(__file__).parents[2] / "local_models" / "Model_Flodelaplace_mocap.osim"
)


def _sample(keypoints: np.ndarray) -> MhrSample:
    return MhrSample(
        verts=np.zeros((3, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=keypoints.astype(np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )


def _model_with_centers(centers_by_joint: dict) -> OsimModel:
    """Model whose joint_centers equal the given points.

    Each joint hangs directly off ground with its parent offset translation set to
    the desired centre, so joint_centers[name] == that translation.
    """
    bodies = []
    joints = []
    for i, (joint_name, center) in enumerate(centers_by_joint.items()):
        child = f"body_{i}"
        bodies.append(OsimBody(child, []))
        joints.append(
            OsimJoint(
                name=joint_name,
                joint_type="CustomJoint",
                parent_body="ground",
                child_body=child,
                parent_offset=OsimFrameOffset(
                    translation=list(center), orientation=[0.0, 0.0, 0.0]
                ),
                child_offset=OsimFrameOffset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                coordinates=[],
            )
        )
    return OsimModel(name="synthetic", bodies=bodies, joints=joints, markers=[])


def test_pairs_are_stable_joints_only():
    # 8 stable pairs: hips, knees, ankles, shoulder centres. Elbows/wrists are
    # excluded because arm pose varies between MHR rest pose and OpenSim neutral.
    assert len(MHR_KP_TO_OSIM_JOINT) == 8
    joints = set(MHR_KP_TO_OSIM_JOINT.values())
    assert joints == {
        "hip_r",
        "hip_l",
        "walker_knee_r",
        "walker_knee_l",
        "ankle_r",
        "ankle_l",
        "acromial_r",
        "acromial_l",
    }
    assert not any("elbow" in j or "radius" in j for j in joints)


def test_build_clouds_pairs_all_present():
    keypoints = np.random.default_rng(0).normal(size=(70, 3))
    centers = {name: keypoints[idx] for idx, name in MHR_KP_TO_OSIM_JOINT.items()}
    source, target, pairs = build_alignment_clouds(
        _sample(keypoints), _model_with_centers(centers)
    )
    assert len(pairs) == len(MHR_KP_TO_OSIM_JOINT)
    assert source.shape == (len(pairs), 3)
    assert target.shape == (len(pairs), 3)
    # source rows are the paired keypoints.
    for row, (kp_idx, _) in zip(source, pairs, strict=True):
        np.testing.assert_allclose(row, keypoints[kp_idx], atol=1e-9)


def test_align_recovers_known_similarity():
    rng = np.random.default_rng(1)
    sample = _sample(rng.normal(size=(70, 3)))
    # Build targets from the *stored* (float32) keypoints so the correspondence is
    # exact in float64 and the recovery residual reflects only solver precision.
    keypoints = np.asarray(sample.keypoints, dtype=float)
    rot = euler_xyz_to_matrix([0.2, -0.5, 0.9])
    scale = 1.7
    trans = np.array([0.3, -0.2, 1.0])
    centers = {
        name: scale * rot @ keypoints[idx] + trans
        for idx, name in MHR_KP_TO_OSIM_JOINT.items()
    }
    model = _model_with_centers(centers)

    transform, residual, pairs = align_mhr_to_opensim(sample, model)

    assert len(pairs) == len(MHR_KP_TO_OSIM_JOINT)
    assert residual < 1e-9
    np.testing.assert_allclose(np.array(transform.rotation), rot, atol=1e-8)
    assert abs(transform.scale - scale) < 1e-8
    np.testing.assert_allclose(transform.translation, trans, atol=1e-8)

    # apply maps source onto target.
    source, target, _ = build_alignment_clouds(sample, model)
    np.testing.assert_allclose(transform.apply(source), target, atol=1e-8)


def test_too_few_pairs_raises():
    keypoints = np.random.default_rng(2).normal(size=(70, 3))
    centers = {  # only 3 joints available -> below MIN_PAIRS
        "hip_r": [0.0, 0.0, 0.0],
        "hip_l": [1.0, 0.0, 0.0],
        "walker_knee_r": [0.0, 1.0, 0.0],
    }
    with pytest.raises(ValueError, match="at least 6"):
        build_alignment_clouds(_sample(keypoints), _model_with_centers(centers))


def test_similarity_to_matrix():
    rot = euler_xyz_to_matrix([0.1, 0.2, 0.3])
    transform, _, _ = align_mhr_to_opensim(
        _sample(np.random.default_rng(3).normal(size=(70, 3))),
        _model_with_centers(
            {
                name: 2.0 * rot @ np.random.default_rng(3).normal(size=(70, 3))[idx]
                for idx, name in MHR_KP_TO_OSIM_JOINT.items()
            }
        ),
    )
    matrix = similarity_to_matrix(transform)
    assert matrix.shape == (4, 4)
    # Top-left block is scale * rotation; bottom row is [0,0,0,1].
    np.testing.assert_allclose(
        matrix[:3, :3], transform.scale * np.array(transform.rotation), atol=1e-9
    )
    np.testing.assert_allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9)


@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_OSIM.exists()),
    reason="real npz/osim not present (CI / clean checkout)",
)
def test_real_alignment():
    from mesh2marker.mhr import load_mhr_npz
    from mesh2marker.osim import parse_osim

    sample = load_mhr_npz(REAL_NPZ)
    model = parse_osim(REAL_OSIM)

    transform, residual, pairs = align_mhr_to_opensim(sample, model)

    # All expected correspondences are found.
    assert len(pairs) == len(MHR_KP_TO_OSIM_JOINT)
    # Plausible scale (real subject vs generic model) and a finite, sane residual.
    assert 0.3 <= transform.scale <= 3.0
    assert np.isfinite(residual)
    assert residual < 0.15

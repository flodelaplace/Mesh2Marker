"""Procrustes pre-alignment of an MHR mesh onto an OpenSim model.

Pure core: stdlib + numpy only. No bpy, no pydantic.

Pairs MHR-70 keypoints (the ``mhr70`` layout) to OpenSim joint centres of the
``Pose2Sim_Wholebody`` model, then solves a similarity transform (Umeyama, via
:mod:`mesh2marker.procrustes`) mapping the MHR camera-frame keypoint cloud onto the
OpenSim neutral-pose joint-centre cloud. This is the coarse pre-alignment; it is not
a per-vertex fit.
"""

from __future__ import annotations

import numpy as np

from .kinematics import joint_centers
from .mhr import MhrSample
from .osim import OsimModel
from .procrustes import SimilarityTransform, procrustes_align

# MHR-70 keypoint index -> Pose2Sim_Wholebody joint name. Keypoint indices follow
# the mhr70 layout; joint names are those exposed by OsimModel for the
# Pose2Sim_Wholebody model.
MHR_KP_TO_OSIM_JOINT: dict[int, str] = {
    10: "hip_r",
    9: "hip_l",
    12: "walker_knee_r",
    11: "walker_knee_l",
    14: "ankle_r",
    13: "ankle_l",
    6: "acromial_r",
    5: "acromial_l",
    8: "elbow_r",
    7: "elbow_l",
}

# Procrustes needs enough non-degenerate correspondences for a stable fit.
MIN_PAIRS = 6


def build_alignment_clouds(
    sample: MhrSample, model: OsimModel
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """Build the (source, target) clouds from paired keypoints / joint centres.

    Source rows are MHR keypoints (camera frame); target rows are the matching
    OpenSim joint centres (neutral pose). Only pairs whose keypoint index is valid
    and whose joint exists in the model are kept. Returns ``(source Nx3, target
    Nx3, pairs)``; raises :class:`ValueError` if fewer than :data:`MIN_PAIRS` pairs
    are available.
    """
    centers = joint_centers(model)
    n_kp = sample.keypoints.shape[0]

    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    pairs: list[tuple[int, str]] = []
    for kp_idx, joint_name in MHR_KP_TO_OSIM_JOINT.items():
        if kp_idx < 0 or kp_idx >= n_kp:
            continue
        if joint_name not in centers:
            continue
        source.append(np.asarray(sample.keypoints[kp_idx], dtype=float))
        target.append(np.asarray(centers[joint_name], dtype=float))
        pairs.append((kp_idx, joint_name))

    if len(pairs) < MIN_PAIRS:
        raise ValueError(
            f"need at least {MIN_PAIRS} keypoint/joint pairs for alignment, "
            f"got {len(pairs)}"
        )
    return np.asarray(source, dtype=float), np.asarray(target, dtype=float), pairs


def align_mhr_to_opensim(
    sample: MhrSample, model: OsimModel, with_scale: bool = True
) -> tuple[SimilarityTransform, float, list[tuple[int, str]]]:
    """Solve the MHR -> OpenSim similarity transform and report the fit residual.

    Returns ``(transform, residual_rms, pairs)`` where ``residual_rms`` is the RMS
    of ``|| transform.apply(source) - target ||`` over the paired points, in metres.
    """
    source, target, pairs = build_alignment_clouds(sample, model)
    transform = procrustes_align(source, target, with_scale=with_scale)
    aligned = transform.apply(source)
    residual_rms = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return transform, residual_rms, pairs


def similarity_to_matrix(transform: SimilarityTransform) -> np.ndarray:
    """4x4 homogeneous matrix for ``y = s R x + t`` (top-left block is ``s * R``)."""
    matrix = np.eye(4)
    matrix[:3, :3] = transform.scale * np.asarray(transform.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(transform.translation, dtype=float)
    return matrix

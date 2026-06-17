"""Per-segment alignment of the OpenSim skeleton onto the MHR mesh.

Pure core: stdlib + numpy only. No bpy, no pydantic.

A single global similarity (see :mod:`mesh2marker.alignment`) places the lower body
and trunk well but leaves a residual gap and cannot follow limbs whose pose differs
(the MHR rest pose has arms raised in an A-pose, the OpenSim neutral pose has arms
down). No global rigid transform fixes that. Instead we align each long bone between
its two joint centres: knowing both joint centres on each side (MHR keypoints via
mhr70, OpenSim centres via :func:`mesh2marker.kinematics.joint_centers`), we fit a
rigid+scale transform per bone so its two ends land on the two corresponding MHR
joints. The bone then sits under the right place of the mesh and the limbs follow.

The mesh itself stays at its global-pre-alignment position; the bones move onto it.
Picking records the MHR vertex INDEX (fixed topology), so it stays exact regardless
of this alignment, which only serves the visual bone/skin judgement.

Note (expected, not a bug): a long bone follows its joint-to-joint axis on the mesh;
the roll is inherited from the neutral pose, so a small residual longitudinal
rotation on heavily-repositioned limbs (arms) is normal.
"""

from __future__ import annotations

import numpy as np

from .alignment import align_mhr_to_opensim, similarity_to_matrix
from .kinematics import forward_kinematics, joint_centers
from .mhr import MhrSample
from .osim import OsimModel
from .procrustes import SimilarityTransform, procrustes_align

# body -> (proximal OpenSim joint, distal OpenSim joint, proximal MHR kp, distal kp)
SEGMENT_TABLE: dict[str, tuple[str, str, int, int]] = {
    "femur_r": ("hip_r", "walker_knee_r", 10, 12),
    "femur_l": ("hip_l", "walker_knee_l", 9, 11),
    "tibia_r": ("walker_knee_r", "ankle_r", 12, 14),
    "tibia_l": ("walker_knee_l", "ankle_l", 11, 13),
    "humerus_r": ("acromial_r", "elbow_r", 6, 8),
    "humerus_l": ("acromial_l", "elbow_l", 5, 7),
    # Forearm: ulna and radius share the elbow -> wrist segment.
    "ulna_r": ("elbow_r", "radius_hand_r", 8, 41),
    "radius_r": ("elbow_r", "radius_hand_r", 8, 41),
    "ulna_l": ("elbow_l", "radius_hand_l", 7, 62),
    "radius_l": ("elbow_l", "radius_hand_l", 7, 62),
}

# body -> ancestor body whose correction it inherits.
INHERIT: dict[str, str] = {
    "patella_r": "femur_r",
    "patella_l": "femur_l",
    "talus_r": "tibia_r",
    "calcn_r": "tibia_r",
    "toes_r": "tibia_r",
    "talus_l": "tibia_l",
    "calcn_l": "tibia_l",
    "toes_l": "tibia_l",
}

# Bodies aligned by a full Procrustes fit on >=3 landmark pairs (MHR keypoint <->
# OpenSim marker). This recovers the full orientation (roll included), which the
# 2-point long-bone fit cannot, and constrains the head (an extremity with no distal
# joint). Each entry: body -> [(MHR keypoint index, OpenSim marker name), ...].
LANDMARK_SEGMENTS: dict[str, list[tuple[int, str]]] = {
    "head": [
        (0, "Nose"),
        (1, "LEye"),
        (2, "REye"),
        (3, "LEar"),
        (4, "REar"),
    ],
    "hand_r": [
        (41, "RWrist_hand"),
        (24, "RThumb"),
        (28, "RIndex"),
        (40, "RPinky"),
        (25, "RIndexTip"),
        (37, "RPinkyTip"),
    ],
    "hand_l": [
        (62, "LWrist_hand"),
        (45, "LThumb"),
        (49, "LIndex"),
        (61, "LPinky"),
        (46, "LIndexTip"),
        (58, "LPinkyTip"),
    ],
}

# Procrustes needs at least 3 non-degenerate correspondences.
LANDMARK_MIN_PAIRS = 3


def minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal proper rotation (det +1) sending unit vector ``a`` onto ``b``.

    Handles ``a`` parallel to ``b`` (identity) and antiparallel (180 degrees about
    an arbitrary axis perpendicular to ``a``). Inputs are normalized defensively.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    cos = float(np.dot(a, b))
    sin = float(np.linalg.norm(v))

    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        # Antiparallel: 180 degrees about any axis perpendicular to a.
        axis = _perpendicular(a)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + cos))


def _perpendicular(a: np.ndarray) -> np.ndarray:
    other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, other)
    return axis / np.linalg.norm(axis)


def _long_bone_matrix(
    c_p: np.ndarray, c_d: np.ndarray, t_p: np.ndarray, t_d: np.ndarray
) -> np.ndarray | None:
    """4x4 similarity sending c_p->t_p and c_d->t_d exactly (minimal rotation)."""
    c_vec = c_d - c_p
    t_vec = t_d - t_p
    c_len = float(np.linalg.norm(c_vec))
    t_len = float(np.linalg.norm(t_vec))
    if c_len < 1e-9 or t_len < 1e-9:
        return None
    scale = t_len / c_len
    rot = minimal_rotation(c_vec / c_len, t_vec / t_len)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = t_p - scale * rot @ c_p
    return matrix


def compute_segment_transforms(
    sample: MhrSample,
    model: OsimModel,
    global_transform: SimilarityTransform | None = None,
) -> dict[str, np.ndarray]:
    """One per-body similarity correction (4x4, OpenSim world frame) for every body.

    Long bones are fit end-to-end (joint centre to joint centre); extremities listed
    in :data:`LANDMARK_SEGMENTS` (head, hands) get a full Procrustes fit on >=3
    landmark pairs, which recovers the roll the 2-point fit cannot and overrides any
    inherited/identity value; remaining listed children inherit their ancestor's
    correction; the trunk and any uncovered body get the identity. ``global_transform``
    maps MHR keypoints into the OpenSim world frame; if omitted it is computed via
    :func:`align_mhr_to_opensim`.
    """
    if global_transform is None:
        global_transform, _, _ = align_mhr_to_opensim(sample, model)

    centers = joint_centers(model)
    n_kp = sample.keypoints.shape[0]

    def kp_world(idx: int) -> np.ndarray:
        point = np.asarray(sample.keypoints[idx], dtype=float)
        return np.asarray(global_transform.apply(point), dtype=float)

    transforms: dict[str, np.ndarray] = {}

    for body, (joint_p, joint_d, kp_p, kp_d) in SEGMENT_TABLE.items():
        if joint_p not in centers or joint_d not in centers:
            continue
        if not (0 <= kp_p < n_kp and 0 <= kp_d < n_kp):
            continue
        matrix = _long_bone_matrix(
            np.asarray(centers[joint_p], dtype=float),
            np.asarray(centers[joint_d], dtype=float),
            kp_world(kp_p),
            kp_world(kp_d),
        )
        if matrix is not None:
            transforms[body] = matrix

    # Full Procrustes for extremities (head, hands) on landmark pairs. These
    # override any inherited/identity value for the listed bodies.
    world = forward_kinematics(model)
    marker_world: dict[str, np.ndarray] = {}
    for marker in model.markers:
        body_world = world.get(marker.parent_body)
        if body_world is None:
            continue
        loc = np.asarray(marker.location, dtype=float)
        marker_world[marker.name] = body_world.rotation @ loc + body_world.translation

    for body, pairs in LANDMARK_SEGMENTS.items():
        source: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for kp_idx, marker_name in pairs:
            if marker_name not in marker_world:
                continue
            if not (0 <= kp_idx < n_kp):
                continue
            source.append(marker_world[marker_name])
            target.append(kp_world(kp_idx))
        if len(source) >= LANDMARK_MIN_PAIRS:
            fit = procrustes_align(
                np.asarray(source, dtype=float),
                np.asarray(target, dtype=float),
                with_scale=True,
            )
            transforms[body] = similarity_to_matrix(fit)

    for body, source_body in INHERIT.items():
        if source_body in transforms:
            transforms[body] = transforms[source_body].copy()

    # Every body of the model gets a correction; trunk and uncovered -> identity.
    for body in model.bodies:
        transforms.setdefault(body.name, np.eye(4))

    return transforms

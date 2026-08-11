"""Per-segment alignment of the OpenSim skeleton onto the MHR mesh.

Pure core: stdlib + numpy only. No bpy, no pydantic.

A single global similarity (see :mod:`mesh2marker.alignment`) places the lower body
and trunk well but leaves a residual gap and cannot follow limbs whose pose differs
(the MHR rest pose has arms raised in an A-pose, the OpenSim neutral pose has arms
down). No global rigid transform fixes that. Instead we align each long bone between
its two joint centres: knowing both joint centres on each side (MHR rig joints from
``sample.joint_coords``, OpenSim centres via
:func:`mesh2marker.kinematics.joint_centers`), we fit a rigid+scale transform per
bone so its two ends land on the two corresponding MHR joints. The bone then sits
under the right place of the mesh and the limbs follow.
Rig joints (not the 70 keypoints) carry the scale block of the extended shape basis,
so bones scale with the subject's full morphology, not the identity alone.

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
from .kinematics import joint_centers
from .markers import neutral_marker_world_positions
from .mhr import MhrSample
from .osim import OsimModel
from .procrustes import SimilarityTransform, procrustes_align

# body -> (proximal OpenSim joint, distal OpenSim joint, proximal MHR joint, distal)
# The MHR-side indices are into the 127-joint rig (``sample.joint_coords``), NOT the
# 70 keypoints. Rig joints carry the scale block of the extended shape basis
# (``dJ[45:73]`` is filled, ``dKP[45:73]`` is zero), so long bones scale with the
# subject's full morphology -- not the identity-only scale the keypoints would give.
# Joints also avoid two keypoints (wrist, ankle) that degenerate at the rest pose.
# Indices validated by the upstream pipeline (left/right symmetry, plausible lengths).
SEGMENT_TABLE: dict[str, tuple[str, str, int, int]] = {
    "femur_r": ("hip_r", "walker_knee_r", 18, 19),
    "femur_l": ("hip_l", "walker_knee_l", 2, 3),
    # Ankle = rig joint 20/4 (kinematically knee+1, coincident with the ankle
    # keypoint at rest). The pipeline's "ankle" 24/8 is ~15 cm more distal (the
    # foot/toe joint) and tilted the tibia ~20 deg forward, out of the mesh.
    "tibia_r": ("walker_knee_r", "ankle_r", 19, 20),
    "tibia_l": ("walker_knee_l", "ankle_l", 3, 4),
    "humerus_r": ("acromial_r", "elbow_r", 39, 40),
    "humerus_l": ("acromial_l", "elbow_l", 75, 76),
    # Forearm: ulna and radius share the elbow -> wrist segment.
    "ulna_r": ("elbow_r", "radius_hand_r", 40, 41),
    "radius_r": ("elbow_r", "radius_hand_r", 40, 41),
    "ulna_l": ("elbow_l", "radius_hand_l", 76, 77),
    "radius_l": ("elbow_l", "radius_hand_l", 76, 77),
}

# Forearm bones: the elbow->wrist axis fit fixes the axis but leaves the roll about
# it (pronation/supination) undetermined -- inherited from the neutral OpenSim pose,
# so the wrist and hand can arrive rotated (~90 deg) versus the MHR, throwing off the
# wrist (RWRA/RWRB) and forearm technical (RFRM/LFRM) markers. When the correspondence
# map provides the two wrist markers, their medial-lateral direction (A = radial
# styloid, B = ulnar styloid) recovers the missing roll: rotate the forearm about its
# own long axis so the OpenSim A->B direction matches the MHR mapped vertices'
# direction. Elbow and bone length are untouched (both joints lie on the axis line).
# ulna and radius share the elbow->wrist fit, so the pair is rigid under it and one
# roll is exact for both. Map-driven only -> the validated Pose2Sim path is untouched.
FOREARM_ROLL_REFERENCE: dict[str, tuple[str, str]] = {
    "ulna_r": ("RWRA", "RWRB"),
    "radius_r": ("RWRA", "RWRB"),
    "ulna_l": ("LWRA", "LWRB"),
    "radius_l": ("LWRA", "LWRB"),
}

# Markers on the forearm that mark the (humeral) epicondyle at the elbow. The
# epicondyle does not pronate, so these must NOT ride the forearm roll -- they are
# repositioned from the forearm's PRE-roll frame (compute_segment_transforms with
# apply_forearm_roll=False). Without this they swing several cm about the long axis.
ELBOW_ROLL_EXEMPT_MARKERS: frozenset[str] = frozenset({"RELB", "LELB"})

# Bodies kept RIGID with an ancestor for the map-driven fit: their mapped markers are
# pooled into the ancestor's Procrustes and they receive its transform, so the shared
# joint cannot tear open. Empty by default: pooling the toes into the foot removes the
# mtp gap and the toe tipping, but it also flattens the toe segment's own orientation,
# which measured WORSE on the plantar seating for the Pose2Sim foot. Kept as the hook
# for models where a short distal segment carries markers only at its far end.
RIGID_WITH: dict[str, str] = {"toes_r": "calcn_r", "toes_l": "calcn_l"}

# Bodies whose hardcoded LANDMARK_SEGMENTS fit is SUPERSEDED when the correspondence
# map covers them: the user's picked vertices beat the fixed mhr70 keypoints.
MAP_OVERRIDES: frozenset[str] = frozenset({"hand_r", "hand_l"})

# Bodies pinned to their PARENT's placement of the shared joint. This is the reverse
# of the junction sewing below (which pulls a parent toward its child): here the child
# is pulled onto a parent that is confidently placed. The hand is fit on its own few
# markers with a free scale, which left it 5.0 cm from where the forearm ends -- the
# wrist visibly came apart. Pinning it closes that to ~1.2 cm while keeping the
# fingers on the skin (making it fully rigid with the forearm instead costs 7 cm at
# the fingers: unlike the foot, the hand has a genuine orientation of its own).
PIN_TO_PARENT_JOINT: frozenset[str] = frozenset({"hand_r", "hand_l"})

# body -> ancestor body whose correction it inherits. Order matters: a child listed
# after its source picks up whatever that source ended up with (fit or inherited).
# The toes follow the FOOT (their parent across the mtp joint), not the shin: when the
# foot is fit on its own markers and the toes inherited the tibia instead, the toes
# bone was left dislocated at the mtp joint (measured: 0.5 cm on the Pose2Sim map,
# 2.9 cm on the SKEL map, where it also dragged the RTOE/RFPI IK markers 3-4 cm).
INHERIT: dict[str, str] = {
    "patella_r": "femur_r",
    "patella_l": "femur_l",
    "talus_r": "tibia_r",
    "calcn_r": "tibia_r",
    "toes_r": "calcn_r",
    "talus_l": "tibia_l",
    "calcn_l": "tibia_l",
    "toes_l": "calcn_l",
    "sacrum": "pelvis",  # the sacrum follows the pelvis
}

# Bodies aligned by a full Procrustes fit on >=3 landmark pairs. This recovers the
# full orientation (roll) and scales volumetric segments (pelvis, torso) in 3D,
# which the 1D long-bone fit cannot, and constrains the head (an extremity with no
# distal joint). Each pair is (source_kind, source_index, OpenSim marker name) with
# source_kind in {"keypoint", "vertex"}: the MHR-side landmark is a pose keypoint
# (mhr70) or a fixed-topology mesh vertex (used for pelvis skin landmarks).
# Vertex indices are known anatomical landmarks of the MHR template mesh.
# lumbar1..5 and Abdomen are intentionally left at identity: they are deep, not
# critical for the bone/skin display and have no reliable surface landmark.
LANDMARK_SEGMENTS: dict[str, list[tuple[str, int, str]]] = {
    "head": [
        ("keypoint", 0, "Nose"),
        ("keypoint", 1, "LEye"),
        ("keypoint", 2, "REye"),
        ("keypoint", 3, "LEar"),
        ("keypoint", 4, "REar"),
    ],
    "hand_r": [
        ("keypoint", 41, "RWrist_hand"),
        ("keypoint", 24, "RThumb"),
        ("keypoint", 28, "RIndex"),
        ("keypoint", 40, "RPinky"),
        ("keypoint", 25, "RIndexTip"),
        ("keypoint", 37, "RPinkyTip"),
    ],
    "hand_l": [
        ("keypoint", 62, "LWrist_hand"),
        ("keypoint", 45, "LThumb"),
        ("keypoint", 49, "LIndex"),
        ("keypoint", 61, "LPinky"),
        ("keypoint", 46, "LIndexTip"),
        ("keypoint", 58, "LPinkyTip"),
    ],
    # Pelvis: skin landmarks (ASIS/PSIS) are mesh vertices, not pose keypoints.
    "pelvis": [
        ("vertex", 7696, "RASI"),
        ("vertex", 6531, "LASI"),
        ("vertex", 7319, "RPSI"),
        ("vertex", 6216, "LPSI"),
    ],
    # Torso: ALL-skin-vertex anchors so the cloud scales COHERENTLY with the subject's
    # morphology. The former keypoint anchors (RACR/LACR/c_neck) do NOT carry the scale
    # block of the extended basis (dKP[45:73]==0) while C7 (a vertex) does, so under a
    # scale morph the mixed cloud distorted and the Procrustes rotated the torso forward
    # (XIPH drifted ~20 cm, c_spine2's X flipped sign). Six well-distributed vertices --
    # lateral acromia, posterior C7 + c_spine2, anterior clavicles -- also break the
    # top-band coplanarity that made the fit ill-conditioned. Indices are fixed MHR
    # topology (same anatomy for every subject).
    "torso": [
        ("vertex", 7878, "RACR"),
        ("vertex", 6620, "LACR"),
        ("vertex", 5768, "C7"),
        ("vertex", 5753, "c_spine2"),
        ("vertex", 7022, "RCLAV"),
        ("vertex", 5787, "LCLAV"),
    ],
}

# Procrustes needs at least 3 non-degenerate correspondences.
LANDMARK_MIN_PAIRS = 3

# Anterior (world +X) shift applied to a map-fit head so the skull's occipital mesh,
# which extends behind the back-of-head markers, seats inside the MHR head. Tunable;
# ~2 cm tucks the occiput while the facial mesh (with anterior slack) stays inside.
HEAD_ANTERIOR_NUDGE_M = 0.02


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


def _rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation (3x3) of angle ``theta`` about unit-normalized ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _roll_matrix_about_axis(
    matrix: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    u_dir: np.ndarray,
    v_dir: np.ndarray,
) -> np.ndarray:
    """``matrix`` rolled about ``axis`` (via ``pivot``) sending ``u_dir`` to ``v_dir``.

    The extra rotation is about the bone's long ``axis`` and pivots on ``pivot`` (the
    proximal joint), so both joint centres -- which lie on the axis line -- stay fixed;
    only the roll (pronation) changes. ``u_dir`` is the bone's current transverse
    direction (after the axis fit), ``v_dir`` the target (MHR). Both are projected onto
    the plane perpendicular to ``axis`` before the signed angle is taken. Returns
    ``matrix`` unchanged if either transverse direction is degenerate (near-parallel to
    the axis).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u_perp = np.asarray(u_dir, dtype=float) - np.dot(u_dir, axis) * axis
    v_perp = np.asarray(v_dir, dtype=float) - np.dot(v_dir, axis) * axis
    if np.linalg.norm(u_perp) < 1e-6 or np.linalg.norm(v_perp) < 1e-6:
        return matrix
    theta = float(
        np.arctan2(np.dot(axis, np.cross(u_perp, v_perp)), np.dot(u_perp, v_perp))
    )
    roll = _rotation_about_axis(axis, theta)
    out = np.eye(4)
    out[:3, :3] = roll @ matrix[:3, :3]
    out[:3, 3] = roll @ (matrix[:3, 3] - pivot) + pivot
    return out


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
    landmark_map: dict[str, int] | None = None,
    apply_forearm_roll: bool = True,
) -> dict[str, np.ndarray]:
    """One per-body similarity correction (4x4, OpenSim world frame) for every body.

    Long bones are fit end-to-end (joint centre to joint centre); bodies listed in
    :data:`LANDMARK_SEGMENTS` (head, hands, and the volumetric pelvis and torso) get
    a full Procrustes fit on >=3 landmark pairs, which recovers the roll the 2-point
    fit cannot and scales volumetric segments in 3D, overriding any inherited/identity
    value; remaining listed children inherit their ancestor's correction; the trunk
    and any uncovered body get the identity. ``global_transform`` maps MHR landmarks
    into the OpenSim world frame; if omitted it is computed via
    :func:`align_mhr_to_opensim`.

    ``landmark_map`` (marker name -> MHR vertex index, e.g. a loaded correspondence)
    is optional and model-agnostic: any body NOT already fit by the long-bone or the
    hardcoded landmark step, but carrying >=3 of the map's markers, is fit by a
    Procrustes on those (source = neutral marker world, target = mapped skin vertex).
    This is how an arbitrary model whose marker names differ from the Pose2Sim
    vocabulary (e.g. the SKEL / bsm marker set) still gets its head, hands, pelvis and
    torso aligned. With ``landmark_map=None`` the behaviour is exactly the legacy one.
    """
    if global_transform is None:
        global_transform, _, _ = align_mhr_to_opensim(sample, model)

    centers = joint_centers(model)
    n_kp = sample.keypoints.shape[0]
    n_verts = sample.verts.shape[0]
    n_joints = sample.joint_coords.shape[0]

    def joint_world(idx: int) -> np.ndarray:
        point = np.asarray(sample.joint_coords[idx], dtype=float)
        return np.asarray(global_transform.apply(point), dtype=float)

    def mhr_world(source_kind: str, idx: int) -> np.ndarray | None:
        """MHR landmark (keypoint or mesh vertex) lifted into the OpenSim frame."""
        if source_kind == "keypoint":
            if not (0 <= idx < n_kp):
                return None
            point = np.asarray(sample.keypoints[idx], dtype=float)
        else:  # "vertex"
            if not (0 <= idx < n_verts):
                return None
            point = np.asarray(sample.verts[idx], dtype=float)
        return np.asarray(global_transform.apply(point), dtype=float)

    transforms: dict[str, np.ndarray] = {}
    marker_world = neutral_marker_world_positions(model)

    for body, (joint_p, joint_d, mhr_p, mhr_d) in SEGMENT_TABLE.items():
        if joint_p not in centers or joint_d not in centers:
            continue
        if not (0 <= mhr_p < n_joints and 0 <= mhr_d < n_joints):
            continue
        t_p = joint_world(mhr_p)
        t_d = joint_world(mhr_d)
        matrix = _long_bone_matrix(
            np.asarray(centers[joint_p], dtype=float),
            np.asarray(centers[joint_d], dtype=float),
            t_p,
            t_d,
        )
        if matrix is None:
            continue
        # Recover the forearm roll from the mapped wrist markers (see
        # FOREARM_ROLL_REFERENCE): the axis fit above leaves it undetermined.
        ref = FOREARM_ROLL_REFERENCE.get(body)
        if landmark_map and ref is not None and apply_forearm_roll:
            mk_a, mk_b = ref
            v_a = landmark_map.get(mk_a)
            v_b = landmark_map.get(mk_b)
            if (
                v_a is not None
                and v_b is not None
                and mk_a in marker_world
                and mk_b in marker_world
            ):
                tgt_a = mhr_world("vertex", int(v_a))
                tgt_b = mhr_world("vertex", int(v_b))
                if tgt_a is not None and tgt_b is not None:
                    u_dir = matrix[:3, :3] @ (marker_world[mk_b] - marker_world[mk_a])
                    matrix = _roll_matrix_about_axis(
                        matrix, t_p, t_d - t_p, u_dir, tgt_b - tgt_a
                    )
        transforms[body] = matrix

    # Full Procrustes on landmark pairs for extremities (head, hands) and volumetric
    # segments (pelvis, torso). These override any inherited/identity value.
    for body, pairs in LANDMARK_SEGMENTS.items():
        source: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for source_kind, source_index, marker_name in pairs:
            if marker_name not in marker_world:
                continue
            mhr_point = mhr_world(source_kind, source_index)
            if mhr_point is None:
                continue
            source.append(marker_world[marker_name])
            target.append(mhr_point)
        if len(source) >= LANDMARK_MIN_PAIRS:
            fit = procrustes_align(
                np.asarray(source, dtype=float),
                np.asarray(target, dtype=float),
                with_scale=True,
            )
            transforms[body] = similarity_to_matrix(fit)

    # Map-driven fits (model-agnostic): fill any body not yet covered above but
    # carrying >=3 of the correspondence map's markers. Runs before INHERIT so a
    # map-fit ancestor (e.g. pelvis) propagates to its children (e.g. sacrum).
    if landmark_map:
        body_of: dict[str, str] = {}
        for mk in model.markers:
            body_of.setdefault(mk.name, mk.parent_body)
        grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for marker_name, vertex_index in landmark_map.items():
            body = body_of.get(marker_name)
            body = RIGID_WITH.get(body, body)  # pool a rigid child into its parent
            if body is None or marker_name not in marker_world:
                continue
            if body in transforms and body not in MAP_OVERRIDES:
                continue
            target = mhr_world("vertex", int(vertex_index))
            if target is None:
                continue
            grouped.setdefault(body, []).append((marker_world[marker_name], target))

        # Scale of a map-fit body comes from the BONE (the global keypoint->joint
        # similarity), NOT from the fitted skin markers. Otherwise obesity -- which
        # widens the skin but not the skeleton -- would inflate the segment's scale
        # (e.g. pelvis 1.3 -> 1.86) and get absorbed there instead of pushing the
        # markers outward. With the bone scale fixed, fat morphs leave the bone size
        # alone and the markers move out onto the skin (the deterministic 5c vision);
        # genuine size changes still scale because the joints (hence global scale)
        # move. Rotation + translation still come from the landmark fit.
        bone_scale = float(global_transform.scale)

        def _fit_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
            src = np.asarray([s for s, _ in pairs], dtype=float)
            tgt = np.asarray([t for _, t in pairs], dtype=float)
            fitted = similarity_to_matrix(procrustes_align(src, tgt, with_scale=True))
            rot = fitted[:3, :3] / np.linalg.norm(fitted[:3, :3], axis=0).mean()
            out = np.eye(4)
            out[:3, :3] = bone_scale * rot
            out[:3, 3] = tgt.mean(axis=0) - bone_scale * (rot @ src.mean(axis=0))
            return out

        map_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for body, pairs in grouped.items():
            if len(pairs) >= LANDMARK_MIN_PAIRS:
                map_pairs[body] = pairs
                transforms[body] = _fit_pairs(pairs)

        # Rigid children take their ancestor's transform, so the shared joint cannot
        # tear open (their markers already fed that ancestor's fit above).
        for child, ancestor in RIGID_WITH.items():
            if ancestor in transforms:
                transforms[child] = transforms[ancestor].copy()

        # Pin a PIN_TO_PARENT_JOINT body onto its parent's placement of their shared
        # joint: refit it with that joint added as an extra pair, weighted like a third
        # of its markers (same idiom as the junction sewing below).
        parent_of = {jt.child_body: (jt.parent_body, jt.name) for jt in model.joints}
        for child in list(map_pairs):
            if child not in PIN_TO_PARENT_JOINT:
                continue
            info = parent_of.get(child)
            if info is None:
                continue
            parent, joint_name = info
            if parent not in transforms or joint_name not in centers:
                continue
            jn = np.asarray(centers[joint_name], dtype=float)
            parent_m = transforms[parent]
            anchor = (jn, parent_m[:3, :3] @ jn + parent_m[:3, 3])
            weight = max(1, round(len(map_pairs[child]) / 3))
            transforms[child] = _fit_pairs(map_pairs[child] + [anchor] * weight)

        # A map-fit head sits perfectly by its markers (front/back/top all on the
        # skin), but the skull MESH extends posterior to its back-of-head markers
        # (LBHD/RBHD), so the occiput pokes out behind the MHR head. The facial mesh
        # has anterior slack (skin sits ahead of the facial bone), so nudging the
        # head bone anteriorly (world +X) seats the occiput inside without the face
        # poking through. Exported markers are unaffected (re-derived from the vertex);
        # only the head-bone frame shifts. Map-driven only -> Pose2Sim head untouched.
        if "head" in map_pairs:
            transforms["head"][0, 3] += HEAD_ANTERIOR_NUDGE_M

        # Sew each map-fit body to its map-fit parent at their shared joint: a
        # well-placed child pulls the parent's junction into place. Concretely the
        # head (fit on its own markers) pulls the torso's neck end forward, so the
        # cervical spine -- rigidly part of the torso -- stops protruding backward.
        # NB: deliberately only map-fit children. The long bones are fit from the rig
        # joints, whose shoulder AP centre reads posterior of the shoulder mass (~39%
        # vs ~51%), so sewing the torso onto them would pull the well-placed glenoid
        # backward. The limb-head placement is governed by the rig joint upstream.
        parent_joint = {jt.child_body: (jt.parent_body, jt.name) for jt in model.joints}
        junctions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for child in map_pairs:
            info = parent_joint.get(child)
            if info is None:
                continue
            parent, joint_name = info
            if parent not in map_pairs or joint_name not in centers:
                continue
            jn = np.asarray(centers[joint_name], dtype=float)
            child_m = transforms[child]
            placed = child_m[:3, :3] @ jn + child_m[:3, 3]
            junctions.setdefault(parent, []).append((jn, placed))
        for parent, extra in junctions.items():
            weight = max(1, round(len(map_pairs[parent]) / 3))
            transforms[parent] = _fit_pairs(map_pairs[parent] + extra * weight)

    for body, source_body in INHERIT.items():
        if body not in transforms and source_body in transforms:
            transforms[body] = transforms[source_body].copy()

    # Every body of the model gets a correction; trunk and uncovered -> identity.
    for body in model.bodies:
        transforms.setdefault(body.name, np.eye(4))

    return transforms

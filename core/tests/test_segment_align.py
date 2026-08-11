"""Per-segment skeleton alignment: minimal_rotation, synthetic exact, real model."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.alignment import align_mhr_to_opensim
from mesh2marker.kinematics import euler_xyz_to_matrix, joint_centers
from mesh2marker.markers import neutral_marker_world_positions
from mesh2marker.mhr import MhrSample
from mesh2marker.osim import (
    OsimBody,
    OsimFrameOffset,
    OsimJoint,
    OsimMarker,
    OsimModel,
)
from mesh2marker.procrustes import SimilarityTransform
from mesh2marker.segment_align import (
    INHERIT,
    LANDMARK_SEGMENTS,
    MAP_OVERRIDES,
    PIN_TO_PARENT_JOINT,
    SEGMENT_TABLE,
    _roll_matrix_about_axis,
    _rotation_about_axis,
    compute_segment_transforms,
    minimal_rotation,
)

REAL_NPZ = Path(__file__).parents[2] / "local_models" / "markers_Squat_mesh.npz"
REAL_OSIM = (
    Path(__file__).parents[2] / "local_models" / "Model_Flodelaplace_mocap.osim"
)
EXT_BASIS = (
    Path(__file__).parents[2] / "local_models" / "mhr_shape_basis_extended.npz"
)
# Live scale directions of the extended basis ([45:73] minus the 4 dead PCA rows).
_DEAD_SCALE = {45, 46, 47, 61}


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


def _sample_with_joints(joints: dict) -> MhrSample:
    joint_coords = np.zeros((127, 3), dtype=np.float32)
    for idx, value in joints.items():
        joint_coords[idx] = value
    return MhrSample(
        verts=np.zeros((3, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=joint_coords,
        keypoints=np.zeros((70, 3), dtype=np.float32),
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


def test_long_bone_maps_joint_centers_onto_rig_joints():
    model = _femur_test_model()
    centers = joint_centers(model)
    j_p = [1.0, 2.0, 3.0]
    j_d = [1.0, 1.0, 3.0]
    # femur_r is fit between rig joints 18 (right_hip) and 19 (right_knee).
    sample = _sample_with_joints({18: j_p, 19: j_d})

    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )

    femur = transforms["femur_r"]
    # global transform is identity, so targets == the raw rig joints.
    np.testing.assert_allclose(_apply(femur, centers["hip_r"]), j_p, atol=1e-9)
    np.testing.assert_allclose(_apply(femur, centers["walker_knee_r"]), j_d, atol=1e-9)

    # Trunk receives the identity.
    np.testing.assert_allclose(transforms["pelvis"], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(transforms["torso"], np.eye(4), atol=1e-12)

    # Inheritance: patella copies the femur correction.
    np.testing.assert_array_equal(transforms["patella_r"], femur)


def _head_model(marker_positions: dict) -> OsimModel:
    # head hangs off ground with identity offset, so world[head] is identity and a
    # marker's world position equals its location.
    body = OsimBody("head", [])
    joint = _joint("neck", "ground", "head", [0, 0, 0])
    markers = [
        OsimMarker(name, "head", list(pos)) for name, pos in marker_positions.items()
    ]
    return OsimModel(name="headmodel", bodies=[body], joints=[joint], markers=markers)


def test_head_landmark_recovers_similarity():
    rng = np.random.default_rng(7)
    names = ["Nose", "LEye", "REye", "LEar", "REar"]
    indices = [0, 1, 2, 3, 4]
    points = {name: rng.normal(size=3) for name in names}
    rot = euler_xyz_to_matrix([0.3, -0.4, 0.7])
    scale = 1.3
    trans = np.array([0.2, -0.1, 0.5])
    kps = {
        idx: scale * rot @ points[name] + trans
        for idx, name in zip(indices, names, strict=True)
    }

    sample = _sample_with_kps(kps)
    model = _head_model(points)
    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )

    head = transforms["head"]
    for idx, name in zip(indices, names, strict=True):
        target = np.asarray(
            _identity_similarity().apply(np.asarray(sample.keypoints[idx], dtype=float))
        )
        np.testing.assert_allclose(_apply(head, points[name]), target, atol=1e-6)


def test_head_landmark_skipped_below_min_pairs():
    rng = np.random.default_rng(8)
    points = {"Nose": rng.normal(size=3), "LEye": rng.normal(size=3)}  # only 2 markers
    sample = _sample_with_kps({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
    model = _head_model(points)
    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )
    # Fewer than 3 pairs: head is not overwritten, stays identity.
    np.testing.assert_allclose(transforms["head"], np.eye(4), atol=1e-12)


def _body_model(body_name: str, markers: list[OsimMarker]) -> OsimModel:
    body = OsimBody(body_name, [])
    joint = _joint(f"{body_name}_jnt", "ground", body_name, [0, 0, 0])
    return OsimModel(name="m", bodies=[body], joints=[joint], markers=markers)


def test_pelvis_landmark_from_vertices_recovers_similarity():
    rng = np.random.default_rng(11)
    pairs = LANDMARK_SEGMENTS["pelvis"]  # all ("vertex", idx, marker)
    max_idx = max(idx for _, idx, _ in pairs)
    verts = np.zeros((max_idx + 1, 3), dtype=np.float32)
    for _, idx, _ in pairs:
        verts[idx] = rng.normal(size=3)

    rot = euler_xyz_to_matrix([0.2, 0.5, -0.3])
    scale = 1.4
    trans = np.array([0.1, 0.3, -0.2])
    # marker world position = similarity(MHR vertex), using the stored float32 vertex.
    markers = [
        OsimMarker(
            name, "pelvis", list(scale * rot @ np.asarray(verts[idx], float) + trans)
        )
        for _, idx, name in pairs
    ]
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )
    model = _body_model("pelvis", markers)

    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )

    pelvis = transforms["pelvis"]
    marker_world = neutral_marker_world_positions(model)
    for _, idx, name in pairs:
        target = np.asarray(verts[idx], dtype=float)  # global transform is identity
        got = _apply(pelvis, marker_world[name])
        np.testing.assert_allclose(got, target, atol=1e-6)


def test_volumetric_landmark_skipped_below_min_pairs():
    rng = np.random.default_rng(12)
    pairs = LANDMARK_SEGMENTS["pelvis"]
    max_idx = max(idx for _, idx, _ in pairs)
    verts = np.zeros((max_idx + 1, 3), dtype=np.float32)
    for _, idx, _ in pairs:
        verts[idx] = rng.normal(size=3)
    # Only 2 of the 4 pelvis markers exist -> 2 valid pairs < 3 -> not overwritten.
    markers = [
        OsimMarker(pairs[0][2], "pelvis", list(verts[pairs[0][1]])),
        OsimMarker(pairs[1][2], "pelvis", list(verts[pairs[1][1]])),
    ]
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )
    model = _body_model("pelvis", markers)
    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity()
    )
    np.testing.assert_allclose(transforms["pelvis"], np.eye(4), atol=1e-12)


def test_every_body_has_a_correction():
    model = _femur_test_model()
    sample = _sample_with_joints({18: [1.0, 2.0, 3.0], 19: [1.0, 1.0, 3.0]})
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

    # Each long bone maps its joint centres exactly onto the MHR rig joints.
    for body, (joint_p, joint_d, mhr_p, mhr_d) in SEGMENT_TABLE.items():
        matrix = transforms[body]
        t_p = np.asarray(
            global_transform.apply(np.asarray(sample.joint_coords[mhr_p], dtype=float))
        )
        t_d = np.asarray(
            global_transform.apply(np.asarray(sample.joint_coords[mhr_d], dtype=float))
        )
        np.testing.assert_allclose(_apply(matrix, centers[joint_p]), t_p, atol=1e-9)
        np.testing.assert_allclose(_apply(matrix, centers[joint_d]), t_d, atol=1e-9)

    # All 30 bodies have a correction.
    assert {b.name for b in model.bodies} <= set(transforms)

    # Inheritance is wired correctly.
    for child, source in INHERIT.items():
        np.testing.assert_array_equal(transforms[child], transforms[source])


@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_OSIM.exists()),
    reason="real npz/osim not present (CI / clean checkout)",
)
def test_real_landmark_extremities():
    from mesh2marker.mhr import load_mhr_npz
    from mesh2marker.osim import parse_osim

    sample = load_mhr_npz(REAL_NPZ)
    model = parse_osim(REAL_OSIM)
    global_transform, _, _ = align_mhr_to_opensim(sample, model)
    transforms = compute_segment_transforms(sample, model, global_transform)

    landmark_bodies = ("head", "hand_r", "hand_l", "pelvis", "torso")

    # Extremities and volumetric segments get a non-identity Procrustes correction.
    for body in landmark_bodies:
        assert body in transforms
        assert not np.allclose(transforms[body], np.eye(4))

    # Hands are no longer mere copies of the forearm (radius) transform.
    assert not np.array_equal(transforms["hand_r"], transforms["radius_r"])
    assert not np.array_equal(transforms["hand_l"], transforms["radius_l"])

    # The sacrum inherits the pelvis correction.
    np.testing.assert_array_equal(transforms["sacrum"], transforms["pelvis"])

    # Per-body Procrustes residual is finite and reasonable.
    marker_world = neutral_marker_world_positions(model)
    n_kp = sample.keypoints.shape[0]
    n_verts = sample.verts.shape[0]

    def mhr_world(source_kind, idx):
        arr = sample.keypoints if source_kind == "keypoint" else sample.verts
        return np.asarray(global_transform.apply(np.asarray(arr[idx], dtype=float)))

    for body in landmark_bodies:
        matrix = transforms[body]
        source = []
        target = []
        for source_kind, idx, marker_name in LANDMARK_SEGMENTS[body]:
            in_bounds = idx < (n_kp if source_kind == "keypoint" else n_verts)
            if marker_name in marker_world and in_bounds:
                source.append(marker_world[marker_name])
                target.append(mhr_world(source_kind, idx))
        source = np.asarray(source)
        target = np.asarray(target)
        assert len(source) >= 3
        aligned = (matrix[:3, :3] @ source.T).T + matrix[:3, 3]
        residual = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
        assert np.isfinite(residual)
        assert residual < 0.1


_LONG_BONES = (
    "femur_r",
    "femur_l",
    "tibia_r",
    "tibia_l",
    "humerus_r",
    "humerus_l",
    "radius_r",
    "radius_l",
    "ulna_r",
    "ulna_l",
)
_LR_PAIRS = (
    ("femur_r", "femur_l"),
    ("tibia_r", "tibia_l"),
    ("humerus_r", "humerus_l"),
    ("radius_r", "radius_l"),
    ("ulna_r", "ulna_l"),
)


# Anatomical reference: the mhr70 keypoint pair bounding each long bone. The rig-joint
# axis used for the fit must agree with it at the rest pose -- guards against a
# mis-identified joint index (the foot/toe joint once tilted the tibia ~20 deg out of
# the mesh).
_KP_REFERENCE = {
    "femur_r": (10, 12),
    "femur_l": (9, 11),
    "tibia_r": (12, 14),
    "tibia_l": (11, 13),
    "humerus_r": (6, 8),
    "humerus_l": (5, 7),
    "radius_r": (8, 41),
    "radius_l": (7, 62),
    "ulna_r": (8, 41),
    "ulna_l": (7, 62),
}


def _unit_axis(p, q) -> np.ndarray:
    v = np.asarray(q, float) - np.asarray(p, float)
    return v / np.linalg.norm(v)


@pytest.mark.skipif(not EXT_BASIS.exists(), reason="extended basis not present")
def test_long_bone_joint_axes_align_with_keypoint_axes_at_rest():
    from mesh2marker.morph import load_shape_basis, morph
    from mesh2marker.segment_align import SEGMENT_TABLE

    sample = morph(load_shape_basis(EXT_BASIS), [0.0] * 73)
    for body, (kp_p, kp_d) in _KP_REFERENCE.items():
        _, _, j_p, j_d = SEGMENT_TABLE[body]
        kp_axis = _unit_axis(sample.keypoints[kp_p], sample.keypoints[kp_d])
        j_axis = _unit_axis(sample.joint_coords[j_p], sample.joint_coords[j_d])
        angle = np.degrees(np.arccos(np.clip(float(kp_axis @ j_axis), -1.0, 1.0)))
        assert angle < 6.0, f"{body} joint axis off by {angle:.1f} deg from keypoints"


def test_landmark_map_fits_an_otherwise_identity_body():
    # A body not covered by SEGMENT_TABLE / LANDMARK_SEGMENTS / INHERIT stays identity
    # without a map, but a landmark_map providing >=3 of its markers fits it (Procrustes
    # marker-world -> mapped skin vertex). This is the model-agnostic path used for the
    # SKEL / bsm marker set, whose names are not in the Pose2Sim vocabulary.
    local = {"M0": [0.0, 0.0, 0.0], "M1": [0.1, 0.0, 0.0], "M2": [0.0, 0.1, 0.0]}
    markers = [OsimMarker(n, "B", v) for n, v in local.items()]  # "B" is in no table
    model = _body_model("B", markers)  # world[B] = identity
    # RIGID target (rotation + translation, NO scale): the map-driven fit takes its
    # scale from the bone (global scale = 1 here, identity), so a rigid target lands
    # exactly. A scaled target would NOT -- by design, fat must not scale the bone.
    rot = euler_xyz_to_matrix([0.2, -0.1, 0.3])
    trans = np.array([0.4, -0.2, 0.5])
    verts = np.array([rot @ np.array(v) + trans for v in local.values()])
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )
    landmark_map = {"M0": 0, "M1": 1, "M2": 2}

    without = compute_segment_transforms(sample, model, _identity_similarity())
    withmap = compute_segment_transforms(
        sample, model, _identity_similarity(), landmark_map=landmark_map
    )

    np.testing.assert_allclose(without["B"], np.eye(4), atol=1e-12)
    assert not np.allclose(withmap["B"], np.eye(4))

    # Each marker lands on its mapped vertex (global transform is identity).
    marker_world = neutral_marker_world_positions(model)
    matrix = withmap["B"]
    for name, idx in landmark_map.items():
        got = matrix[:3, :3] @ marker_world[name] + matrix[:3, 3]
        np.testing.assert_allclose(got, verts[idx], atol=1e-6)


def test_map_sewing_pulls_parent_junction_toward_child():
    # A map-fit child (e.g. head) sews its parent (torso) at their shared joint, so the
    # parent's placement of that joint moves toward the child's -- the mechanism that
    # pulls the torso's neck end forward. Verify the junction gap shrinks vs a plain
    # parent-only Procrustes.
    from mesh2marker.procrustes import procrustes_align
    from mesh2marker.segment_align import similarity_to_matrix

    # torso (parent) hangs off ground; head (child) hangs off torso via head_torso.
    tm = {f"T{i}": v for i, v in enumerate([[0.1, 0, 0], [0, 0, 0.1], [-0.1, 0.05, 0]])}
    hm = {
        f"H{i}": v
        for i, v in enumerate([[0.05, 0.6, 0], [0, 0.6, 0.05], [0, 0.55, 0]])
    }
    markers = [OsimMarker(n, "torso", v) for n, v in tm.items()]
    markers += [OsimMarker(n, "head", v) for n, v in hm.items()]
    bodies = [OsimBody("torso", []), OsimBody("head", [])]
    joints = [
        _joint("ground_torso", "ground", "torso", [0, 0, 0]),
        _joint("head_torso", "torso", "head", [0, 0.5, 0]),
    ]
    model = OsimModel(name="m", bodies=bodies, joints=joints, markers=markers)

    # Targets: torso markers stay put (torso ~identity); head markers shift +0.12 in X.
    shift = np.array([0.12, 0.0, 0.0])
    verts = []
    order = list(tm) + list(hm)
    for n in order:
        base = np.array(tm.get(n, hm.get(n)), dtype=float)
        verts.append(base + (shift if n in hm else 0.0))
    verts = np.array(verts)
    landmark_map = {n: i for i, n in enumerate(order)}
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )

    transforms = compute_segment_transforms(
        sample, model, _identity_similarity(), landmark_map=landmark_map
    )
    joint = np.array([0.0, 0.5, 0.0])  # head_torso centre (identity global)

    def place(matrix):
        return matrix[:3, :3] @ joint + matrix[:3, 3]

    head_at = place(transforms["head"])
    gap_sewn = float(np.linalg.norm(place(transforms["torso"]) - head_at))

    # Plain torso-only fit (no sewing) for the baseline gap.
    src = np.array([tm[n] for n in tm], dtype=float)
    tgt = np.array([verts[landmark_map[n]] for n in tm], dtype=float)
    plain = similarity_to_matrix(procrustes_align(src, tgt, with_scale=True))
    gap_plain = float(np.linalg.norm(place(plain) - head_at))

    assert gap_sewn < gap_plain  # sewing closes the junction


def test_torso_anchors_are_all_skin_vertices():
    # All torso anchors must be skin vertices: they carry the scale block of the
    # extended basis, so the cloud scales coherently. Mixing in keypoints (whose scale
    # block is zero) distorted the cloud under a scale morph and rotated the torso
    # forward (XIPH drifted ~20 cm, c_spine2's X flipped sign).
    kinds = {kind for kind, _, _ in LANDMARK_SEGMENTS["torso"]}
    assert kinds == {"vertex"}, kinds
    assert len(LANDMARK_SEGMENTS["torso"]) >= 4  # well-distributed, not the top band


@pytest.mark.skipif(
    not (EXT_BASIS.exists() and REAL_OSIM.exists()),
    reason="extended basis / real osim not present",
)
def test_torso_rotation_stable_under_scale_morph():
    # Regression for the torso-forward-rotation bug: the torso segment rotation must be
    # nearly identical at rest and under a strong scale morph.
    from mesh2marker.morph import load_shape_basis, morph
    from mesh2marker.osim import parse_osim

    basis = load_shape_basis(EXT_BASIS)
    model = parse_osim(REAL_OSIM)

    def torso_rotation(betas):
        sample = morph(basis, betas)
        gt, _, _ = align_mhr_to_opensim(sample, model)
        m = compute_segment_transforms(sample, model, gt)["torso"]
        r = m[:3, :3]
        return r / np.linalg.norm(r, axis=0)  # strip the (uniform) scale

    r_rest = torso_rotation([0.0] * 73)
    scaled = [0.0] * 73
    for i in range(45, 73):
        if i not in _DEAD_SCALE:
            scaled[i] = 1.5
    r_scaled = torso_rotation(scaled)

    angle = np.degrees(
        np.arccos(np.clip((np.trace(r_rest.T @ r_scaled) - 1.0) / 2.0, -1.0, 1.0))
    )
    assert angle < 3.0, f"torso rotated {angle:.1f} deg under a scale-only morph"


def _bone_scale(transforms: dict, body: str) -> float:
    # The linear part of a long-bone correction is scale * rotation, so the mean
    # column norm recovers the per-segment scale factor.
    return float(np.linalg.norm(transforms[body][:3, :3], axis=0).mean())


@pytest.mark.skipif(
    not (EXT_BASIS.exists() and REAL_OSIM.exists()),
    reason="extended basis / real osim not present",
)
def test_long_bones_scale_with_extended_basis_scale_block():
    # Regression for the keypoint->rig-joint switch. Rig joints carry dJ[45:73], so
    # the scale block now changes long-bone lengths; with keypoints (dKP[45:73]==0)
    # every long bone stayed at exactly the identity scale (0% change).
    from mesh2marker.morph import load_shape_basis, morph
    from mesh2marker.osim import parse_osim

    basis = load_shape_basis(EXT_BASIS)
    model = parse_osim(REAL_OSIM)

    def bone_scales(betas):
        sample = morph(basis, betas)
        gt, _, _ = align_mhr_to_opensim(sample, model)
        transforms = compute_segment_transforms(sample, model, gt)
        return {body: _bone_scale(transforms, body) for body in _LONG_BONES}

    identity_only = [0.0] * 73
    identity_only[5] = 1.0
    with_scale = list(identity_only)
    for i in range(45, 73):  # every live scale direction active
        if i not in _DEAD_SCALE:
            with_scale[i] = 1.0

    base = bone_scales(identity_only)
    scaled = bone_scales(with_scale)

    # Every long bone now follows the morphology's scale block (was 0% before).
    for body in _LONG_BONES:
        rel = abs(scaled[body] - base[body]) / base[body]
        assert rel > 1e-3, f"{body} did not scale ({rel:.1e})"

    # Left/right symmetry holds under the same (symmetric) scale vector.
    for right, left in _LR_PAIRS:
        np.testing.assert_allclose(scaled[right], scaled[left], rtol=1e-3)


@pytest.mark.skipif(
    not (EXT_BASIS.exists() and REAL_OSIM.exists()),
    reason="extended basis / real osim not present",
)
def test_long_bone_lengths_are_plausible_and_symmetric_at_rest():
    from mesh2marker.morph import load_shape_basis, morph

    basis = load_shape_basis(EXT_BASIS)
    sample = morph(basis, [0.0] * 73)  # rest pose

    from mesh2marker.segment_align import SEGMENT_TABLE

    lengths = {}
    for body, (_, _, mhr_p, mhr_d) in SEGMENT_TABLE.items():
        lengths[body] = float(
            np.linalg.norm(sample.joint_coords[mhr_d] - sample.joint_coords[mhr_p])
        )

    # Adult-plausible long-bone lengths (metres).
    for body in _LONG_BONES:
        assert 0.15 < lengths[body] < 0.55, f"{body} length {lengths[body]:.3f} m"

    # The MHR template is symmetric: left/right rest lengths match to < 1 mm.
    for right, left in _LR_PAIRS:
        assert abs(lengths[right] - lengths[left]) < 1e-3


def test_rotation_about_axis_rotates_in_plane():
    # 90 deg about +Z sends +X -> +Y; the axis direction itself is fixed.
    r = _rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(r @ [0.0, 0.0, 3.0], [0.0, 0.0, 3.0], atol=1e-12)
    # It is a proper rotation.
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
    # Axis need not be pre-normalized.
    r2 = _rotation_about_axis(np.array([0.0, 0.0, 5.0]), np.pi / 2)
    np.testing.assert_allclose(r2, r, atol=1e-12)


def test_roll_matrix_aligns_transverse_and_fixes_the_axis_line():
    # Roll a bone (identity placement) about +Z so its transverse +X meets target +Y.
    axis = np.array([0.0, 0.0, 1.0])
    pivot = np.array([0.0, 0.0, 0.0])
    rolled = _roll_matrix_about_axis(
        np.eye(4), pivot, axis, u_dir=np.array([1.0, 0.0, 0.0]),
        v_dir=np.array([0.0, 2.0, 0.0]),
    )
    # The transverse direction now points along +Y (magnitude is not constrained).
    moved = rolled[:3, :3] @ [1.0, 0.0, 0.0]
    moved /= np.linalg.norm(moved)
    np.testing.assert_allclose(moved, [0.0, 1.0, 0.0], atol=1e-12)
    # Points on the axis line through the pivot are untouched (both bone ends stay put).
    for p in ([0.0, 0.0, 0.0], [0.0, 0.0, 0.42], [0.0, 0.0, -0.17]):
        q = rolled[:3, :3] @ p + rolled[:3, 3]
        np.testing.assert_allclose(q, p, atol=1e-12)


def test_roll_only_uses_the_component_perpendicular_to_the_axis():
    # A target that is purely along the axis carries no roll information -> no change.
    axis = np.array([0.0, 0.0, 1.0])
    unchanged = _roll_matrix_about_axis(
        np.eye(4), np.zeros(3), axis, u_dir=np.array([1.0, 0.0, 0.0]),
        v_dir=np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(unchanged, np.eye(4), atol=1e-12)


def _forearm_model() -> OsimModel:
    # ulna_r hangs off ground at the elbow; radius_hand_r gives the distal (wrist)
    # centre. Elbow/wrist markers sit lateral to the long axis (offset in Z).
    bodies = [OsimBody(n, []) for n in ("ulna_r", "hand_r")]
    joints = [
        _joint("elbow_r", "ground", "ulna_r", [0.1, 1.4, 0.2]),
        _joint("radius_hand_r", "ground", "hand_r", [0.1, 1.0, 0.2]),
    ]
    markers = [
        OsimMarker("RELB", "ulna_r", [0.0, 0.0, 0.1]),   # world (0.1, 1.4, 0.3)
        OsimMarker("RWRA", "ulna_r", [0.0, -0.4, 0.1]),  # world (0.1, 1.0, 0.3)
        OsimMarker("RWRB", "ulna_r", [0.0, -0.4, -0.1]), # world (0.1, 1.0, 0.1)
    ]
    return OsimModel(name="fa", bodies=bodies, joints=joints, markers=markers)


def _forearm_sample() -> MhrSample:
    joint_coords = np.zeros((127, 3), dtype=np.float32)
    joint_coords[40] = [0.1, 1.4, 0.2]  # rig elbow
    joint_coords[41] = [0.1, 1.0, 0.2]  # rig wrist
    verts = np.zeros((3, 3), dtype=np.float32)
    verts[1] = [0.2, 1.0, 0.2]  # RWRA target vertex  -> transverse along -X ...
    verts[2] = [0.0, 1.0, 0.2]  # RWRB target vertex
    return MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=joint_coords,
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )


def test_apply_forearm_roll_toggle_rolls_only_when_enabled():
    model = _forearm_model()
    sample = _forearm_sample()
    lm = {"RWRA": 1, "RWRB": 2}  # marker -> MHR vertex index

    rolled = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity(), landmark_map=lm
    )
    straight = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity(),
        landmark_map=lm, apply_forearm_roll=False,
    )

    # Rig elbow/wrist coincide with the OpenSim joint centres, so the pure axis fit is
    # the identity; disabling the roll must leave ulna_r at that identity ...
    np.testing.assert_allclose(straight["ulna_r"], np.eye(4), atol=1e-6)
    # ... while enabling it rolls the bone about its long axis (target transverse
    # direction differs from the neutral one), so ulna_r is no longer the identity.
    assert not np.allclose(rolled["ulna_r"], np.eye(4), atol=1e-6)
    # A roll is a pure rotation: it preserves the joint centres on the axis line.
    for jc in ([0.1, 1.4, 0.2], [0.1, 1.0, 0.2]):
        p = rolled["ulna_r"][:3, :3] @ jc + rolled["ulna_r"][:3, 3]
        np.testing.assert_allclose(p, jc, atol=1e-6)


def _foot_model() -> OsimModel:
    """tibia -> talus -> calcn -> toes, with three markers to map-fit the calcn."""
    names = ("tibia_r", "talus_r", "calcn_r", "toes_r")
    bodies = [OsimBody(n, []) for n in names]
    joints = [
        _joint("walker_knee_r", "ground", "tibia_r", [0.1, 0.5, 0.08]),
        _joint("ankle_r", "tibia_r", "talus_r", [0.0, -0.4, 0.0]),
        _joint("subtalar_r", "talus_r", "calcn_r", [0.0, -0.04, 0.0]),
        _joint("mtp_r", "calcn_r", "toes_r", [0.15, -0.02, 0.0]),
    ]
    markers = [
        OsimMarker("RCAL", "calcn_r", [-0.02, 0.0, 0.0]),
        OsimMarker("RTOE", "calcn_r", [0.18, 0.0, 0.01]),
        OsimMarker("RMT5", "calcn_r", [0.14, 0.0, 0.05]),
    ]
    return OsimModel(name="foot", bodies=bodies, joints=joints, markers=markers)


def test_toes_follow_the_foot_not_the_shin():
    # The toes' parent across the mtp joint is the calcaneus. Inheriting the tibia
    # instead left the toes bone dislocated at the mtp joint.
    assert INHERIT["toes_r"] == "calcn_r"
    assert INHERIT["toes_l"] == "calcn_l"

    model = _foot_model()
    rng = np.random.default_rng(4)
    verts = rng.normal(size=(30, 3)).astype(np.float32)
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )
    # Map the three calcn markers so the foot gets its own map-driven fit.
    transforms = compute_segment_transforms(
        sample,
        model,
        global_transform=_identity_similarity(),
        landmark_map={"RCAL": 1, "RTOE": 5, "RMT5": 9},
    )

    # The toes ride the foot's correction, not the shin's ...
    np.testing.assert_allclose(transforms["toes_r"], transforms["calcn_r"], atol=1e-12)
    assert not np.allclose(transforms["toes_r"], transforms["tibia_r"], atol=1e-6)
    # ... so the shared mtp joint centre lands in ONE place (no dislocated toe bone).
    mtp = joint_centers(model)["mtp_r"]
    by_foot = _apply(transforms["calcn_r"], mtp)
    by_toes = _apply(transforms["toes_r"], mtp)
    np.testing.assert_allclose(by_foot, by_toes, atol=1e-12)


def _forearm_hand_model() -> OsimModel:
    """radius_r fit end-to-end by SEGMENT_TABLE, hand_r hanging off it at the wrist."""
    bodies = [OsimBody(n, []) for n in ("radius_r", "hand_r")]
    joints = [
        _joint("elbow_r", "ground", "radius_r", [0.1, 1.4, 0.2]),
        _joint("radius_hand_r", "radius_r", "hand_r", [0.0, -0.4, 0.0]),
    ]
    # Three hand markers, offset from the wrist so they can drive a Procrustes.
    markers = [
        OsimMarker("RThumb", "hand_r", [0.02, -0.05, 0.01]),
        OsimMarker("RIndex", "hand_r", [0.0, -0.09, 0.02]),
        OsimMarker("RPinky", "hand_r", [-0.02, -0.08, -0.02]),
    ]
    return OsimModel(name="fh", bodies=bodies, joints=joints, markers=markers)


def test_hand_is_pinned_to_the_wrist_the_forearm_places():
    assert "hand_r" in MAP_OVERRIDES and "hand_r" in PIN_TO_PARENT_JOINT

    model = _forearm_hand_model()
    joint_coords = np.zeros((127, 3), dtype=np.float32)
    joint_coords[40] = [0.1, 1.4, 0.2]  # rig elbow
    joint_coords[41] = [0.1, 1.0, 0.2]  # rig wrist
    rng = np.random.default_rng(3)
    verts = rng.normal(size=(40, 3)).astype(np.float32) * 0.05 + np.array(
        [0.1, 0.95, 0.2], dtype=np.float32
    )
    sample = MhrSample(
        verts=verts,
        faces=np.zeros((1, 3), dtype=np.int32),
        joint_coords=joint_coords,
        keypoints=np.zeros((70, 3), dtype=np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )
    lm = {"RThumb": 1, "RIndex": 5, "RPinky": 9}
    transforms = compute_segment_transforms(
        sample, model, global_transform=_identity_similarity(), landmark_map=lm
    )

    wrist = joint_centers(model)["radius_hand_r"]
    by_forearm = _apply(transforms["radius_r"], wrist)
    by_hand = _apply(transforms["hand_r"], wrist)
    pinned = float(np.linalg.norm(by_forearm - by_hand))

    # Same fit without the pin: the hand follows only its own markers and drifts off
    # the wrist. The pin must bring it measurably closer to where the forearm ends.
    import mesh2marker.segment_align as _sa

    saved = _sa.PIN_TO_PARENT_JOINT
    _sa.PIN_TO_PARENT_JOINT = frozenset()
    try:
        free = compute_segment_transforms(
            sample, model, global_transform=_identity_similarity(), landmark_map=lm
        )
    finally:
        _sa.PIN_TO_PARENT_JOINT = saved
    unpinned = float(np.linalg.norm(by_forearm - _apply(free["hand_r"], wrist)))

    assert pinned < unpinned
    # The hand still has its own orientation: it is NOT simply the forearm's transform.
    assert not np.allclose(transforms["hand_r"], transforms["radius_r"], atol=1e-6)

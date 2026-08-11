"""Stateless subject-osim generation and the headless CLI."""

import json
import xml.etree.ElementTree as ET

import numpy as np

from mesh2marker import cli
from mesh2marker.alignment import align_mhr_to_opensim
from mesh2marker.correspondence import links_to_correspondence
from mesh2marker.generate import generate_subject_osim, sample_from_betas
from mesh2marker.io import write as write_correspondence_file
from mesh2marker.linking import centroid_vertex, reposition_marker_to_vertex
from mesh2marker.mhr import MhrSample
from mesh2marker.osim import parse_osim
from mesh2marker.segment_align import compute_segment_transforms

# The 8 joints used by the per-segment alignment (so align/compute succeed), each
# hanging off ground with a distinct centre.
JOINTS = [
    ("hip_r", "femur_r", (0.1, 0.9, 0.08)),
    ("hip_l", "femur_l", (0.1, 0.9, -0.08)),
    ("walker_knee_r", "tibia_r", (0.1, 0.5, 0.08)),
    ("walker_knee_l", "tibia_l", (0.1, 0.5, -0.08)),
    ("ankle_r", "talus_r", (0.1, 0.1, 0.08)),
    ("ankle_l", "talus_l", (0.1, 0.1, -0.08)),
    ("acromial_r", "humerus_r", (0.1, 1.4, 0.18)),
    ("acromial_l", "humerus_l", (0.1, 1.4, -0.18)),
]


def _write_synthetic_osim(path, markers) -> None:
    """markers: list of (name, parent_body, (x, y, z))."""
    doc = ET.Element("OpenSimDocument", Version="40000")
    model = ET.SubElement(doc, "Model", name="synthetic")

    body_objs = ET.SubElement(ET.SubElement(model, "BodySet"), "objects")
    for _, child_body, _ in JOINTS:
        ET.SubElement(body_objs, "Body", name=child_body)

    joint_objs = ET.SubElement(ET.SubElement(model, "JointSet"), "objects")
    for jname, child_body, centre in JOINTS:
        joint = ET.SubElement(joint_objs, "CustomJoint", name=jname)
        ET.SubElement(joint, "socket_parent_frame").text = f"{jname}_g"
        ET.SubElement(joint, "socket_child_frame").text = f"{jname}_c"
        frames = ET.SubElement(joint, "frames")
        pof_g = ET.SubElement(frames, "PhysicalOffsetFrame", name=f"{jname}_g")
        ET.SubElement(pof_g, "socket_parent").text = "/ground"
        cx, cy, cz = centre
        ET.SubElement(pof_g, "translation").text = f"{cx} {cy} {cz}"
        ET.SubElement(pof_g, "orientation").text = "0 0 0"
        pof_c = ET.SubElement(frames, "PhysicalOffsetFrame", name=f"{jname}_c")
        ET.SubElement(pof_c, "socket_parent").text = f"/bodyset/{child_body}"
        ET.SubElement(pof_c, "translation").text = "0 0 0"
        ET.SubElement(pof_c, "orientation").text = "0 0 0"

    marker_objs = ET.SubElement(ET.SubElement(model, "MarkerSet"), "objects")
    for name, parent_body, loc in markers:
        marker = ET.SubElement(marker_objs, "Marker", name=name)
        ET.SubElement(marker, "socket_parent_frame").text = f"/bodyset/{parent_body}"
        ET.SubElement(marker, "location").text = f"{loc[0]} {loc[1]} {loc[2]}"

    ET.ElementTree(doc).write(path, encoding="UTF-8", xml_declaration=True)


TWO_MARKERS = [
    ("MK1", "femur_r", (0.0, 0.0, 0.0)),
    ("MK2", "humerus_r", (0.0, 0.0, 0.0)),
]


def _make_sample(n_verts=50, seed=0) -> MhrSample:
    rng = np.random.default_rng(seed)
    return MhrSample(
        verts=rng.normal(size=(n_verts, 3)).astype(np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        joint_coords=np.zeros((127, 3), dtype=np.float32),
        keypoints=rng.normal(size=(70, 3)).astype(np.float32),
        frame_index=0,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="synthetic",
    )


def test_generate_subject_osim(tmp_path):
    src = str(tmp_path / "src.osim")
    dst = str(tmp_path / "out.osim")
    _write_synthetic_osim(
        src,
        markers=[
            ("MK1", "femur_r", (0.0, 0.0, 0.0)),
            ("MK2", "humerus_r", (0.0, 0.0, 0.0)),
            ("MK3", "tibia_r", (0.11, 0.22, 0.33)),  # not in the map -> unchanged
        ],
    )
    sample = _make_sample()
    correspondence = {"MK1": [3], "MK2": [7, 8]}

    locations = generate_subject_osim(src, dst, correspondence, sample)

    assert set(locations) == {"MK1", "MK2"}

    # The written locations match a direct recomputation (same chain, same map).
    model = parse_osim(src)
    gt, _, _ = align_mhr_to_opensim(sample, model)
    landmark_map = {
        n: centroid_vertex(sample.verts, list(idx))
        for n, idx in correspondence.items()
        if idx
    }
    seg = compute_segment_transforms(sample, model, gt, landmark_map=landmark_map)
    for name, indices in correspondence.items():
        idx = centroid_vertex(sample.verts, indices)
        expected = reposition_marker_to_vertex(model, name, sample.verts, idx, gt, seg)
        np.testing.assert_allclose(locations[name], expected, atol=1e-9)

    # Output .osim has the new positions; the unmapped marker is untouched.
    out = {m.name: m.location for m in parse_osim(dst).markers}
    np.testing.assert_allclose(out["MK1"], locations["MK1"], atol=1e-9)
    np.testing.assert_allclose(out["MK2"], locations["MK2"], atol=1e-9)
    np.testing.assert_allclose(out["MK3"], [0.11, 0.22, 0.33], atol=1e-9)


def _write_basis(tmp_path, n=50, jn=10, kn=70, s=3) -> str:
    rng = np.random.default_rng(1)
    path = tmp_path / "shape_basis.npz"
    np.savez(
        path,
        V0=rng.normal(size=(n, 3)),
        J0=rng.normal(size=(jn, 3)),
        KP0=rng.normal(size=(kn, 3)),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        dV=rng.normal(size=(s, n, 3)),
        dJ=rng.normal(size=(s, jn, 3)),
        dKP=rng.normal(size=(s, kn, 3)),
        delta=np.float64(0.1),
        meta=json.dumps({"units": "meters", "coordinate_frame": "mhr_rest"}),
    )
    return str(path)


def test_sample_from_betas_zero_is_mean(tmp_path):
    from mesh2marker.morph import load_shape_basis

    basis = load_shape_basis(_write_basis(tmp_path))
    sample = sample_from_betas(basis, np.zeros(3))
    np.testing.assert_allclose(sample.verts, basis.v0, atol=1e-12)


def _write_corr(tmp_path) -> str:
    path = tmp_path / "map.json"
    links = [
        {"marker": "MK1", "vertex_indices": [3], "opensim_body": "femur_r"},
        {"marker": "MK2", "vertex_indices": [7, 8], "opensim_body": "humerus_r"},
    ]
    # Synthetic (non-vocabulary) input map for the plumbing test: build the schema
    # object and write it with the low-level writer, bypassing export validation.
    corr = links_to_correspondence(
        links, mhr_topology_id="mhr-50", opensim_model="synthetic", marker_set="test"
    )
    write_correspondence_file(corr, path)
    return str(path)


def test_cli_generate_with_npz(tmp_path):
    src = str(tmp_path / "src.osim")
    _write_synthetic_osim(src, markers=TWO_MARKERS)
    corr = _write_corr(tmp_path)

    # Write the sample as an npz the loader accepts.
    sample = _make_sample()
    npz = str(tmp_path / "subject.npz")
    np.savez(
        npz,
        verts=sample.verts,
        faces=sample.faces,
        joint_coords=sample.joint_coords,
        keypoints=sample.keypoints,
        coordinate_frame="estimator_camera_raw",
        units="meters",
        source="subject",
    )
    dst = str(tmp_path / "out.osim")

    rc = cli.main(
        ["generate", "--osim", src, "--map", corr, "--output", dst, "--npz", npz]
    )
    assert rc == 0
    names = {m.name for m in parse_osim(dst).markers}
    assert {"MK1", "MK2"} <= names


def test_cli_batch_with_betas_csv(tmp_path):
    src = str(tmp_path / "src.osim")
    _write_synthetic_osim(src, markers=TWO_MARKERS)
    corr = _write_corr(tmp_path)
    basis = _write_basis(tmp_path)

    csv_path = tmp_path / "betas.csv"
    csv_path.write_text("subjA,0.1,0.0,-0.2\nsubjB,0.3,0.2,0.1\n")
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "batch",
            "--osim", src,
            "--map", corr,
            "--basis", basis,
            "--betas-csv", str(csv_path),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "subjA.osim").is_file()
    assert (out_dir / "subjB.osim").is_file()


# JOINTS + a forearm (elbow_r -> radius_hand_r) so the roll engages and RELB is exempt.
_FOREARM_JOINTS = [
    ("elbow_r", "ulna_r", (0.1, 1.4, 0.18)),
    ("radius_hand_r", "hand_r", (0.1, 1.0, 0.18)),
]


def _write_osim_with_forearm(path) -> None:
    doc = ET.Element("OpenSimDocument", Version="40000")
    model = ET.SubElement(doc, "Model", name="fa")
    body_objs = ET.SubElement(ET.SubElement(model, "BodySet"), "objects")
    for _, child_body, _ in (*JOINTS, *_FOREARM_JOINTS):
        ET.SubElement(body_objs, "Body", name=child_body)
    joint_objs = ET.SubElement(ET.SubElement(model, "JointSet"), "objects")
    for jname, child_body, centre in (*JOINTS, *_FOREARM_JOINTS):
        joint = ET.SubElement(joint_objs, "CustomJoint", name=jname)
        ET.SubElement(joint, "socket_parent_frame").text = f"{jname}_g"
        ET.SubElement(joint, "socket_child_frame").text = f"{jname}_c"
        frames = ET.SubElement(joint, "frames")
        pof_g = ET.SubElement(frames, "PhysicalOffsetFrame", name=f"{jname}_g")
        ET.SubElement(pof_g, "socket_parent").text = "/ground"
        cx, cy, cz = centre
        ET.SubElement(pof_g, "translation").text = f"{cx} {cy} {cz}"
        ET.SubElement(pof_g, "orientation").text = "0 0 0"
        pof_c = ET.SubElement(frames, "PhysicalOffsetFrame", name=f"{jname}_c")
        ET.SubElement(pof_c, "socket_parent").text = f"/bodyset/{child_body}"
        ET.SubElement(pof_c, "translation").text = "0 0 0"
        ET.SubElement(pof_c, "orientation").text = "0 0 0"
    marker_objs = ET.SubElement(ET.SubElement(model, "MarkerSet"), "objects")
    for name, loc in (
        ("RELB", (0.0, 0.0, 0.05)),
        ("RWRA", (0.0, -0.4, 0.05)),
        ("RWRB", (0.0, -0.4, -0.05)),
    ):
        marker = ET.SubElement(marker_objs, "Marker", name=name)
        ET.SubElement(marker, "socket_parent_frame").text = "/bodyset/ulna_r"
        ET.SubElement(marker, "location").text = f"{loc[0]} {loc[1]} {loc[2]}"
    ET.ElementTree(doc).write(path, encoding="UTF-8", xml_declaration=True)


def _make_forearm_sample() -> MhrSample:
    sample = _make_sample(n_verts=20, seed=3)
    sample.joint_coords[40] = [0.1, 1.4, 0.18]
    sample.joint_coords[41] = [0.1, 1.0, 0.18]
    return sample


def test_generate_exempts_elbow_markers_from_forearm_roll(tmp_path):
    src = str(tmp_path / "fa.osim")
    dst = str(tmp_path / "out.osim")
    _write_osim_with_forearm(src)
    sample = _make_forearm_sample()
    corr = {"RELB": [0], "RWRA": [1], "RWRB": [2]}

    locations = generate_subject_osim(src, dst, corr, sample)

    model = parse_osim(src)
    gt, _, _ = align_mhr_to_opensim(sample, model)
    lm = {n: centroid_vertex(sample.verts, i) for n, i in corr.items()}
    seg = compute_segment_transforms(sample, model, gt, landmark_map=lm)
    seg_nr = compute_segment_transforms(
        sample, model, gt, landmark_map=lm, apply_forearm_roll=False
    )

    # The elbow marker is placed from the PRE-roll forearm frame ...
    exp_elbow = reposition_marker_to_vertex(
        model, "RELB", sample.verts, centroid_vertex(sample.verts, [0]), gt, seg_nr
    )
    np.testing.assert_allclose(locations["RELB"], exp_elbow, atol=1e-6)
    # ... the wrist marker from the rolled frame.
    exp_wrist = reposition_marker_to_vertex(
        model, "RWRA", sample.verts, centroid_vertex(sample.verts, [1]), gt, seg
    )
    np.testing.assert_allclose(locations["RWRA"], exp_wrist, atol=1e-6)

    # Guard: the roll is actually non-trivial here, so the exemption matters (the elbow
    # would have landed elsewhere had it ridden the roll).
    assert not np.allclose(seg["ulna_r"], seg_nr["ulna_r"], atol=1e-6)
    roll_elbow = reposition_marker_to_vertex(
        model, "RELB", sample.verts, centroid_vertex(sample.verts, [0]), gt, seg
    )
    assert not np.allclose(locations["RELB"], roll_elbow, atol=1e-4)


def _synth_basis_127(seed=5):
    """73-component basis with 127 rig joints; scale block [45:73] moves the joints."""
    from mesh2marker.morph import ShapeBasis

    rng = np.random.default_rng(seed)
    nj, nv, nk, ns = 127, 50, 70, 73
    dj = np.zeros((ns, nj, 3))
    dj[45:] = rng.normal(size=(28, nj, 3)) * 0.1
    return ShapeBasis(
        v0=rng.normal(size=(nv, 3)),
        j0=rng.normal(size=(nj, 3)) + 2.0,
        kp0=rng.normal(size=(nk, 3)),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        dv=rng.normal(size=(ns, nv, 3)),
        dj=dj,
        dkp=np.zeros((ns, nk, 3)),
        delta=0.1,
        meta={},
    )


def test_generate_returns_segment_scales_when_shape_basis_given(tmp_path):
    from mesh2marker.generate import SubjectOsim
    from mesh2marker.morph import morph

    src = str(tmp_path / "src.osim")
    dst = str(tmp_path / "out.osim")
    _write_synthetic_osim(src, markers=TWO_MARKERS)
    basis = _synth_basis_127()
    corr = {"MK1": [3], "MK2": [7, 8]}

    # Default (no shape_basis): the plain {marker: location} dict, unchanged.
    loc = generate_subject_osim(src, dst, corr, morph(basis, np.zeros(73)))
    assert isinstance(loc, dict)

    # Opt-in: SubjectOsim carrying the same markers + a complete scale dict.
    res = generate_subject_osim(
        src, dst, corr, morph(basis, np.zeros(73)), shape_basis=basis
    )
    assert isinstance(res, SubjectOsim)
    assert res.marker_locations == loc  # markers identical to the default path
    model_bodies = {b.name for b in parse_osim(src).bodies}
    assert set(res.segment_scales) == model_bodies  # every body present, none dropped
    for body, s in res.segment_scales.items():  # betas=0 -> all unscaled
        np.testing.assert_allclose(s, (1.0, 1.0, 1.0), atol=1e-9, err_msg=body)

    # A scale morph drives the bones through the subject rig joints.
    betas = np.zeros(73)
    betas[48] = 2.0
    scaled = generate_subject_osim(
        src, dst, corr, morph(basis, betas), shape_basis=basis
    )
    assert not np.allclose(scaled.segment_scales["femur_r"], (1.0, 1.0, 1.0))


def test_alignment_markers_restricts_who_drives_the_fit(tmp_path):
    """Contact-only markers can be excluded from the alignment yet still be placed."""
    src = str(tmp_path / "src.osim")
    dst = str(tmp_path / "out.osim")
    # femur_r carries 3 anatomical markers plus one off-axis "contact" marker.
    _write_synthetic_osim(
        src,
        markers=[
            ("MK1", "femur_r", (0.02, 0.0, 0.0)),
            ("MK2", "femur_r", (0.0, 0.03, 0.0)),
            ("MK3", "femur_r", (0.0, 0.0, 0.04)),
            ("CONTACT", "femur_r", (0.05, -0.06, 0.07)),
        ],
    )
    sample = _make_sample(seed=11)
    corr = {"MK1": [3], "MK2": [7], "MK3": [11], "CONTACT": [19]}
    ik = ["MK1", "MK2", "MK3"]

    restricted = generate_subject_osim(
        src, dst, corr, sample, alignment_markers=ik
    )
    full = generate_subject_osim(src, dst, corr, sample)

    # Every mapped marker is still written, including the excluded one.
    assert set(restricted) == {"MK1", "MK2", "MK3", "CONTACT"}

    # Restricting genuinely changes the fit (otherwise the test proves nothing).
    assert not np.allclose(restricted["MK1"], full["MK1"], atol=1e-6)

    # It matches a direct recomputation whose landmark_map holds only the IK markers.
    model = parse_osim(src)
    gt, _, _ = align_mhr_to_opensim(sample, model)
    lm = {n: centroid_vertex(sample.verts, corr[n]) for n in ik}
    seg = compute_segment_transforms(sample, model, gt, landmark_map=lm)
    for name, indices in corr.items():
        expected = reposition_marker_to_vertex(
            model, name, sample.verts, centroid_vertex(sample.verts, indices), gt, seg
        )
        np.testing.assert_allclose(restricted[name], expected, atol=1e-9)

    # An empty restriction is honoured (no map-driven fit at all), not treated as None.
    none_driving = generate_subject_osim(src, dst, corr, sample, alignment_markers=[])
    assert not np.allclose(none_driving["MK1"], full["MK1"], atol=1e-6)

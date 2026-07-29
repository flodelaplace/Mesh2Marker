"""Per-body skeletal scale factors from the MHR rig joints."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.morph import ShapeBasis, morph
from mesh2marker.segment_scale import (
    SEGMENT_SPANS,
    segment_scales_from_joints,
    subject_segment_scales,
)

EXT_BASIS = Path(__file__).parents[2] / "local_models" / "mhr_shape_basis_extended.npz"


def _synth_basis(seed: int = 0) -> ShapeBasis:
    """73-component basis where ONLY the scale block [45:73] moves the 127 rig joints.

    Mirrors the real extended basis (identity directions leave the joints untouched),
    so the tests exercise the segment-scale logic deterministically without the .npz.
    """
    rng = np.random.default_rng(seed)
    n_joints, n_verts, n_kp, n_shape = 127, 50, 70, 73
    dj = np.zeros((n_shape, n_joints, 3))
    dj[45:] = rng.normal(size=(28, n_joints, 3)) * 0.1
    return ShapeBasis(
        v0=rng.normal(size=(n_verts, 3)),
        j0=rng.normal(size=(n_joints, 3)) + 2.0,
        kp0=rng.normal(size=(n_kp, 3)),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        dv=rng.normal(size=(n_shape, n_verts, 3)),
        dj=dj,
        dkp=np.zeros((n_shape, n_kp, 3)),
        delta=0.1,
        meta={},
    )


def test_zero_betas_is_identity_scale_everywhere():
    scales = subject_segment_scales(_synth_basis(), np.zeros(73))
    assert scales  # non-empty
    for body, s in scales.items():
        np.testing.assert_allclose(s, (1.0, 1.0, 1.0), atol=1e-12, err_msg=body)


def test_identity_betas_do_not_scale_the_skeleton():
    # Fat/shape (identity block) leaves the rig joints -> every scale stays 1.0.
    betas = np.zeros(73)
    betas[0] = 3.0
    for body, s in subject_segment_scales(_synth_basis(), betas).items():
        np.testing.assert_allclose(s, (1.0, 1.0, 1.0), atol=1e-12, err_msg=body)


def test_scale_betas_scale_a_long_bone_by_its_span_ratio():
    basis = _synth_basis()
    betas = np.zeros(73)
    betas[48] = 2.0  # a live scale direction
    scales = subject_segment_scales(basis, betas)
    assert not np.allclose(scales["femur_r"], (1.0, 1.0, 1.0))
    # The factor is exactly the subject/template length ratio of femur_r's span,
    # which guards that the table maps femur_r to the right rig-joint indices.
    js = morph(basis, betas).joint_coords
    p, d = SEGMENT_SPANS["femur_r"]
    ratio = float(
        np.linalg.norm(js[d] - js[p]) / np.linalg.norm(basis.j0[d] - basis.j0[p])
    )
    np.testing.assert_allclose(scales["femur_r"], (ratio, ratio, ratio), atol=1e-9)
    # Isotropic.
    assert scales["femur_r"][0] == scales["femur_r"][1] == scales["femur_r"][2]


def test_inherited_children_follow_their_parent():
    betas = np.zeros(73)
    betas[48] = 2.0
    scales = subject_segment_scales(_synth_basis(), betas)
    np.testing.assert_allclose(scales["patella_r"], scales["femur_r"])
    np.testing.assert_allclose(scales["talus_r"], scales["tibia_r"])
    np.testing.assert_allclose(scales["sacrum"], scales["pelvis"])


def test_bodies_argument_lists_every_body_with_unscaled_as_identity():
    bodies = ["femur_r", "toes_r", "ground", "made_up_body"]
    scales = subject_segment_scales(_synth_basis(), np.zeros(73), bodies=bodies)
    assert set(scales) == set(bodies)  # all present, none dropped
    # Bodies with no span/inheritance are explicitly (1,1,1), not absent.
    np.testing.assert_allclose(scales["made_up_body"], (1.0, 1.0, 1.0))
    np.testing.assert_allclose(scales["ground"], (1.0, 1.0, 1.0))


def test_scales_from_joints_handles_short_joint_arrays():
    # Fewer joints than the largest span index -> those bodies are simply skipped.
    j0 = np.ones((30, 3))
    js = np.ones((30, 3)) * 2.0
    scales = segment_scales_from_joints(j0, js)  # torso span (35,112) out of range
    assert "torso" not in scales


@pytest.mark.skipif(not EXT_BASIS.is_file(), reason="extended basis not available")
def test_real_basis_fat_immune_size_active():
    from mesh2marker.morph import load_shape_basis

    basis = load_shape_basis(str(EXT_BASIS))
    fat = np.zeros(73)
    fat[0] = 3.0
    for body, s in subject_segment_scales(basis, fat).items():
        np.testing.assert_allclose(s, (1.0, 1.0, 1.0), atol=1e-6, err_msg=body)
    size = np.zeros(73)
    size[45:] = 1.0  # turn on the whole scale block
    scales = subject_segment_scales(basis, size)
    assert any(not np.allclose(s, (1.0, 1.0, 1.0), atol=1e-3) for s in scales.values())

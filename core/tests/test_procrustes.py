"""Synthetic, deterministic validation of the Umeyama similarity transform."""

import numpy as np
import pytest

from mesh2marker.models import FrameAlignment
from mesh2marker.procrustes import (
    SimilarityTransform,
    procrustes_align,
    to_frame_alignment,
)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_exact_recovery_with_scale():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 3))
    r_true = _rot_z(0.3) @ _rot_y(-0.5) @ _rot_x(0.8)
    s_true = 2.5
    t_true = np.array([1.0, -2.0, 0.5])
    target = s_true * source @ r_true.T + t_true

    st = procrustes_align(source, target, with_scale=True)
    rot = np.array(st.rotation)

    np.testing.assert_allclose(rot, r_true, atol=1e-9)
    np.testing.assert_allclose(st.translation, t_true, atol=1e-9)
    assert abs(st.scale - s_true) < 1e-9
    np.testing.assert_allclose(st.apply(source), target, atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_rigid_recovers_rotation_and_translation():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(15, 3))
    r_true = _rot_x(0.4) @ _rot_z(-0.7)
    t_true = np.array([0.2, 0.5, -1.0])
    target = source @ r_true.T + t_true  # no scale

    st = procrustes_align(source, target, with_scale=False)

    assert st.scale == 1.0
    np.testing.assert_allclose(np.array(st.rotation), r_true, atol=1e-9)
    np.testing.assert_allclose(st.translation, t_true, atol=1e-9)
    np.testing.assert_allclose(st.apply(source), target, atol=1e-9)


def test_rigid_does_not_absorb_scale():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(15, 3))
    r_true = _rot_y(0.6)
    target = 3.0 * source @ r_true.T  # data genuinely has scale

    st = procrustes_align(source, target, with_scale=False)

    assert st.scale == 1.0
    residual = np.linalg.norm(st.apply(source) - target)
    assert residual > 1e-3  # rigid fit cannot absorb the scale -> non-zero residual


def test_rejects_reflection():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(30, 3))
    mirror = np.diag([1.0, 1.0, -1.0])  # det = -1, an improper map
    target = source @ mirror.T

    # A naive SVD (R = U @ Vt) would return this improper, det = -1 matrix.
    cov = (target - target.mean(0)).T @ (source - source.mean(0)) / source.shape[0]
    u, _, vt = np.linalg.svd(cov)
    assert np.linalg.det(u @ vt) < 0  # the naive solution is a reflection

    st = procrustes_align(source, target, with_scale=True)
    rot = np.array(st.rotation)

    # The returned solution must be a proper rotation, not the reflection.
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.norm(rot - mirror) > 1e-6


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        procrustes_align(np.zeros((10, 3)), np.zeros((9, 3)))


def test_non_3d_raises():
    with pytest.raises(ValueError):
        procrustes_align(np.zeros((10, 2)), np.zeros((10, 2)))


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        procrustes_align(np.zeros((2, 3)), np.zeros((2, 3)))


def test_degenerate_source_raises():
    source = np.ones((5, 3))  # all identical -> zero variance
    target = np.random.default_rng(4).normal(size=(5, 3))
    with pytest.raises(ValueError):
        procrustes_align(source, target)


def test_to_frame_alignment():
    st = SimilarityTransform(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[1.0, 2.0, 3.0],
        scale=2.0,
    )
    fa = to_frame_alignment(st)

    assert isinstance(fa, FrameAlignment)
    assert fa.rotation == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert fa.translation == [1.0, 2.0, 3.0]
    assert fa.scale == 2.0

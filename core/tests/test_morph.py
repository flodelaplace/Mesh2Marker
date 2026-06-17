"""Linear shape morph: load/validate the basis, exactness and linearity."""

import json

import numpy as np
import pytest

from mesh2marker.morph import (
    component_displacements,
    load_shape_basis,
    morph,
)

N, J, K, S = 6, 4, 5, 3
DELTA = 0.1


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "V0": rng.normal(size=(N, 3)),
        "J0": rng.normal(size=(J, 3)),
        "KP0": rng.normal(size=(K, 3)),
        "faces": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        "dV": rng.normal(size=(S, N, 3)),
        "dJ": rng.normal(size=(S, J, 3)),
        "dKP": rng.normal(size=(S, K, 3)),
        "delta": np.float64(DELTA),
        "meta": json.dumps(
            {"units": "meters", "coordinate_frame": "mhr_rest", "n_shape": S}
        ),
    }


def _write(tmp_path, **overrides) -> str:
    data = _arrays()
    data.update(overrides)
    path = tmp_path / "shape_basis.npz"
    np.savez(path, **data)
    return str(path)


def test_load_shape_basis_valid(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    assert basis.v0.shape == (N, 3)
    assert basis.dv.shape == (S, N, 3)
    assert basis.dj.shape == (S, J, 3)
    assert basis.dkp.shape == (S, K, 3)
    assert basis.faces.shape == (2, 3)
    assert basis.n_shape == S
    assert basis.meta["coordinate_frame"] == "mhr_rest"


def test_load_shape_basis_inconsistent_raises(tmp_path):
    bad = np.zeros((S, N + 1, 3))  # dV vertex count mismatches V0
    with pytest.raises(ValueError, match="dV"):
        load_shape_basis(_write(tmp_path, dV=bad))


def test_morph_zero_returns_base(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    sample = morph(basis, np.zeros(S))
    np.testing.assert_allclose(sample.verts, basis.v0, atol=1e-12)
    np.testing.assert_allclose(sample.joint_coords, basis.j0, atol=1e-12)
    np.testing.assert_allclose(sample.keypoints, basis.kp0, atol=1e-12)


def test_morph_single_component(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    betas = np.zeros(S)
    betas[1] = DELTA
    sample = morph(basis, betas)
    np.testing.assert_allclose(sample.verts, basis.v0 + DELTA * basis.dv[1], atol=1e-6)
    np.testing.assert_allclose(
        sample.joint_coords, basis.j0 + DELTA * basis.dj[1], atol=1e-6
    )
    np.testing.assert_allclose(
        sample.keypoints, basis.kp0 + DELTA * basis.dkp[1], atol=1e-6
    )


def test_morph_superposition(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    a = np.array([0.2, -0.1, 0.4])
    b = np.array([-0.3, 0.5, 0.1])
    va = morph(basis, a).verts
    vb = morph(basis, b).verts
    vab = morph(basis, a + b).verts
    np.testing.assert_allclose(vab, va + vb - basis.v0, atol=1e-6)


def test_morph_betas_too_long_raises(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    with pytest.raises(ValueError, match="exceeds n_shape"):
        morph(basis, np.zeros(S + 1))


def test_morph_betas_shorter_zero_padded(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    sample = morph(basis, [DELTA])  # only first component
    np.testing.assert_allclose(sample.verts, basis.v0 + DELTA * basis.dv[0], atol=1e-6)


def test_morph_sample_fields(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    sample = morph(basis, np.zeros(S))
    assert sample.verts.shape == (N, 3)
    assert sample.joint_coords.shape == (J, 3)
    assert sample.keypoints.shape == (K, 3)
    np.testing.assert_array_equal(sample.faces, basis.faces)
    assert sample.coordinate_frame == "mhr_rest"
    assert sample.source == "morph"
    assert sample.units == "meters"
    assert sample.betas == [0.0] * S


def test_component_displacements_length(tmp_path):
    basis = load_shape_basis(_write(tmp_path))
    disp = component_displacements(basis)
    assert disp.shape == (S,)
    assert np.all(disp >= 0)

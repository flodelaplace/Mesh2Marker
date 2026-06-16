"""Loader for MHR .npz samples: round-trip + one failing case per rule."""

from pathlib import Path

import numpy as np
import pytest

from mesh2marker.mhr import load_mhr_npz

REAL_NPZ = (
    Path(__file__).parents[2] / "local_models" / "markers_Squat_mesh.npz"
)


def _base() -> dict:
    """A small but valid sample: 6 verts, 4 faces, 127 joints, 70 keypoints."""
    return {
        "verts": np.random.default_rng(0).random((6, 3)).astype(np.float32),
        "faces": np.array(
            [[0, 1, 2], [1, 2, 3], [3, 4, 5], [0, 2, 4]], dtype=np.int32
        ),
        "joint_coords": np.random.default_rng(1).random((127, 3)),
        "keypoints": np.random.default_rng(2).random((70, 3)),
        "frame_index": 7,
        "coordinate_frame": "opensim_y_up",
        "units": "m",
        "n_vertices": 6,
        "source": "demo_video_opensim",
    }


def _write(tmp_path, data: dict) -> str:
    path = tmp_path / "sample.npz"
    np.savez(path, **data)
    return str(path)


def test_load_roundtrip(tmp_path):
    sample = load_mhr_npz(_write(tmp_path, _base()))

    assert sample.verts.shape == (6, 3)
    assert np.issubdtype(sample.verts.dtype, np.floating)
    assert sample.faces.shape == (4, 3)
    assert np.issubdtype(sample.faces.dtype, np.integer)
    assert sample.joint_coords.shape == (127, 3)
    assert sample.keypoints.shape == (70, 3)

    assert sample.frame_index == 7
    assert sample.coordinate_frame == "opensim_y_up"
    assert sample.units == "m"
    assert sample.source == "demo_video_opensim"


def test_meta_defaults_when_absent(tmp_path):
    data = _base()
    for key in ("frame_index", "coordinate_frame", "units", "source", "n_vertices"):
        data.pop(key)
    sample = load_mhr_npz(_write(tmp_path, data))
    assert sample.frame_index == 0
    assert sample.coordinate_frame == "unknown"
    assert sample.units == "m"
    assert sample.source == "unknown"


def test_faces_out_of_bounds_raises(tmp_path):
    data = _base()
    data["faces"] = np.array([[0, 1, 2], [1, 2, 99]], dtype=np.int32)  # 99 >= 6
    with pytest.raises(ValueError, match="face indices"):
        load_mhr_npz(_write(tmp_path, data))


def test_bad_joint_shape_raises(tmp_path):
    data = _base()
    data["joint_coords"] = np.random.default_rng(3).random((100, 3))
    with pytest.raises(ValueError, match="joint_coords"):
        load_mhr_npz(_write(tmp_path, data))


def test_bad_keypoints_shape_raises(tmp_path):
    data = _base()
    data["keypoints"] = np.random.default_rng(4).random((71, 3))
    with pytest.raises(ValueError, match="keypoints"):
        load_mhr_npz(_write(tmp_path, data))


def test_verts_wrong_shape_raises(tmp_path):
    data = _base()
    data["verts"] = np.random.default_rng(5).random((6, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="verts"):
        load_mhr_npz(_write(tmp_path, data))


def test_faces_not_integer_raises(tmp_path):
    data = _base()
    data["faces"] = data["faces"].astype(np.float32)
    with pytest.raises(ValueError, match="integer"):
        load_mhr_npz(_write(tmp_path, data))


def test_missing_key_raises(tmp_path):
    data = _base()
    del data["keypoints"]
    with pytest.raises(ValueError, match="missing required key"):
        load_mhr_npz(_write(tmp_path, data))


def test_n_vertices_mismatch_raises(tmp_path):
    data = _base()
    data["n_vertices"] = 999
    with pytest.raises(ValueError, match="n_vertices"):
        load_mhr_npz(_write(tmp_path, data))


@pytest.mark.skipif(
    not REAL_NPZ.exists(), reason="real example npz not present (CI / clean checkout)"
)
def test_real_pipeline_sample():
    sample = load_mhr_npz(REAL_NPZ)
    n = sample.verts.shape[0]
    assert sample.verts.shape == (n, 3)
    assert sample.faces.shape[1] == 3
    assert sample.joint_coords.shape == (127, 3)
    assert sample.keypoints.shape == (70, 3)
    # All face indices reference real vertices.
    assert int(sample.faces.min()) >= 0
    assert int(sample.faces.max()) < n

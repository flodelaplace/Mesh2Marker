"""Geometry resolution, transform composition, and Y-up -> Z-up conversion."""

from pathlib import Path

import numpy as np

from mesh2marker.geometry import (
    Y_UP_TO_Z_UP,
    geometry_world_matrix,
    resolve_geometry_file,
)
from mesh2marker.kinematics import Transform
from mesh2marker.osim import OsimFrameOffset, OsimGeometry

REAL_GEOMETRY_DIR = Path(__file__).parents[2] / "local_models" / "Geometry"


def _geom(scale_factors, local_offset=None) -> OsimGeometry:
    return OsimGeometry(
        mesh_name="g",
        mesh_file="g.vtp",
        scale_factors=scale_factors,
        local_offset=local_offset or OsimFrameOffset.identity(),
    )


def _touch(path: Path):
    path.write_bytes(b"")


def test_resolve_vtp_to_stl(tmp_path):
    _touch(tmp_path / "sacrum.stl")
    resolved = resolve_geometry_file("sacrum.vtp", tmp_path)
    assert resolved == tmp_path / "sacrum.stl"


def test_resolve_case_insensitive(tmp_path):
    _touch(tmp_path / "femur_r.STL")
    resolved = resolve_geometry_file("Femur_R.vtp", tmp_path)
    assert resolved is not None
    assert resolved.name == "femur_r.STL"


def test_resolve_prefers_stl_over_obj(tmp_path):
    _touch(tmp_path / "pelvis.stl")
    _touch(tmp_path / "pelvis.obj")
    resolved = resolve_geometry_file("pelvis.vtp", tmp_path)
    assert resolved == tmp_path / "pelvis.stl"


def test_resolve_falls_back_to_obj(tmp_path):
    _touch(tmp_path / "tibia.obj")
    resolved = resolve_geometry_file("tibia.vtp", tmp_path)
    assert resolved == tmp_path / "tibia.obj"


def test_resolve_missing_returns_none(tmp_path):
    assert resolve_geometry_file("nope.vtp", tmp_path) is None


def test_resolve_missing_dir_returns_none(tmp_path):
    assert resolve_geometry_file("sacrum.vtp", tmp_path / "does_not_exist") is None


def test_geometry_world_matrix_translation_and_scale():
    world = Transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
    matrix = geometry_world_matrix(world, _geom([2.0, 2.0, 2.0]))
    point = np.array([1.0, 0.0, 0.0, 1.0])
    # scale (2,2,2) then identity offset then translate (1,2,3): (3, 2, 3).
    np.testing.assert_allclose(matrix @ point, [3.0, 2.0, 3.0, 1.0], atol=1e-9)


def test_geometry_world_matrix_rotation_order():
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = Transform(rz90, np.zeros(3))
    matrix = geometry_world_matrix(world, _geom([1.0, 1.0, 1.0]))
    point = np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix @ point, [0.0, 1.0, 0.0, 1.0], atol=1e-9)


def test_geometry_world_matrix_empty_scale_is_unit():
    world = Transform(np.eye(3), np.zeros(3))
    matrix = geometry_world_matrix(world, _geom([]))
    np.testing.assert_allclose(matrix, np.eye(4), atol=1e-9)


def test_geometry_world_matrix_with_local_offset():
    # world: rotate +90 about Z, translate (1,0,0).
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = Transform(rz90, np.array([1.0, 0.0, 0.0]))
    # geometry: scale 2, local offset translation (1,0,0), no offset rotation.
    offset = OsimFrameOffset(translation=[1.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0])
    matrix = geometry_world_matrix(world, _geom([2.0, 2.0, 2.0], offset))

    # Hand computation for local point p = (1, 0, 0):
    #   scale -> (2, 0, 0); offset translate -> (3, 0, 0);
    #   world: Rz90 @ (3,0,0) = (0, 3, 0), + (1,0,0) = (1, 3, 0).
    point = np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix @ point, [1.0, 3.0, 0.0, 1.0], atol=1e-9)


def test_identity_offset_matches_plain_world_scale():
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = Transform(rz90, np.array([0.5, -1.0, 2.0]))
    geom = _geom([1.3, 0.7, 2.1])
    # Identity offset => world_body @ scale (the previous behaviour).
    expected = np.eye(4)
    expected[:3, :3] = world.rotation
    expected[:3, 3] = world.translation
    expected = expected @ np.diag([1.3, 0.7, 2.1, 1.0])
    np.testing.assert_allclose(geometry_world_matrix(world, geom), expected, atol=1e-9)


def test_y_up_to_z_up_sends_y_to_z():
    y_axis = np.array([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(Y_UP_TO_Z_UP @ y_axis, [0.0, 0.0, 1.0, 1.0], atol=1e-9)
    # Proper rotation.
    np.testing.assert_allclose(np.linalg.det(Y_UP_TO_Z_UP[:3, :3]), 1.0, atol=1e-9)


def test_real_geometry_dir_resolves_vtp_to_stl():
    if not REAL_GEOMETRY_DIR.is_dir():
        return  # optional; absent in CI / clean checkout
    # Every .vtp that has a sibling .stl must resolve to that .stl.
    vtps = list(REAL_GEOMETRY_DIR.glob("*.vtp"))
    assert vtps, "expected .vtp files in the real Geometry dir"
    resolved_any = False
    for vtp in vtps:
        result = resolve_geometry_file(vtp.name, REAL_GEOMETRY_DIR)
        if (REAL_GEOMETRY_DIR / f"{vtp.stem}.stl").is_file():
            assert result == REAL_GEOMETRY_DIR / f"{vtp.stem}.stl"
            resolved_any = True
    assert resolved_any

"""Resolve OpenSim segment geometry files and compose their world placement.

Pure core: stdlib + numpy only. No bpy. The bpy layer imports the resolved
``.stl`` files and applies the 4x4 matrices computed here; all path resolution and
transform composition live here.

The model references ``.vtp`` meshes, but on disk we ship ``.stl`` (loadable
without vtk). Geometries live in the body's local frame and carry per-axis
``scale_factors``; :func:`geometry_world_matrix` composes scale then the body world
transform. :data:`Y_UP_TO_Z_UP` converts the whole OpenSim (Y-up) model to
Blender's (Z-up) convention and is meant to be applied once, globally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .kinematics import Transform

# Preference order when several on-disk formats exist for the same mesh stem.
_GEOMETRY_EXTENSIONS = (".stl", ".obj")

# Rotation of +90 degrees about X. Maps OpenSim Y-up to Blender Z-up:
# it sends (0, 1, 0) -> (0, 0, 1). Apply once to the whole model.
Y_UP_TO_Z_UP: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def resolve_geometry_file(mesh_file: str, geometry_dir: str | Path) -> Path | None:
    """Find the on-disk geometry for a model ``mesh_file`` reference.

    The model references e.g. ``sacrum.vtp``; we look for a file with the same
    stem and a usable extension (``.stl`` preferred, then ``.obj``) in
    ``geometry_dir``. Matching is case-insensitive on both stem and extension.
    Returns ``None`` if nothing matches (a body may simply have no available
    geometry); it never raises for a missing file.
    """
    geometry_dir = Path(geometry_dir)
    if not geometry_dir.is_dir():
        return None

    stem = Path(mesh_file).stem

    # Fast path: exact stem with a preferred extension.
    for ext in _GEOMETRY_EXTENSIONS:
        candidate = geometry_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    # Case-insensitive fallback, keeping the extension preference order.
    stem_lower = stem.lower()
    matches: dict[str, Path] = {}
    for entry in geometry_dir.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in _GEOMETRY_EXTENSIONS and entry.stem.lower() == stem_lower:
            matches.setdefault(ext, entry)
    for ext in _GEOMETRY_EXTENSIONS:
        if ext in matches:
            return matches[ext]
    return None


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Convert a :class:`~mesh2marker.kinematics.Transform` to a 4x4 matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = transform.rotation
    matrix[:3, 3] = transform.translation
    return matrix


def geometry_world_matrix(
    world_body: Transform, scale_factors: list[float] | np.ndarray | None
) -> np.ndarray:
    """Full 4x4 placement of a geometry: ``world_body ∘ scale(scale_factors)``.

    Order: a local geometry point ``p`` is first scaled (per axis, in the body's
    local frame), then mapped by the body world transform. As a matrix product
    ``M = world_body_4x4 @ scale_4x4`` and ``p_world = M @ [p; 1]``. An empty or
    missing ``scale_factors`` defaults to unit scale; any other non-length-3 value
    raises :class:`ValueError`.
    """
    if scale_factors is None or len(scale_factors) == 0:
        scale = np.ones(3)
    else:
        scale = np.asarray(scale_factors, dtype=float).reshape(-1)
        if scale.size != 3:
            raise ValueError(
                f"scale_factors must have length 3, got {len(scale_factors)}"
            )
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    return transform_to_matrix(world_body) @ scale_matrix

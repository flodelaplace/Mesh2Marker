"""Load an MHR (Momentum Human Rig) mesh sample from a ``.npz`` file.

Pure core: stdlib + numpy only (numpy ships inside Blender's embedded Python, so
this respects the zero-wheel rule). No bpy, no pydantic.

This loader reads the native cloud as-is: no axis/units conversion happens here.
The coordinate frame is recorded in :attr:`MhrSample.coordinate_frame`; converting
it is a display / alignment concern, not a loading concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_JOINT_COORDS = 127
N_KEYPOINTS = 70


@dataclass(eq=False)
class MhrSample:
    """One MHR sample: the fixed-topology mesh plus joints/keypoints and metadata."""

    verts: np.ndarray  # (N, 3) float
    faces: np.ndarray  # (F, 3) int
    joint_coords: np.ndarray  # (127, 3) float
    keypoints: np.ndarray  # (70, 3) float
    frame_index: int
    coordinate_frame: str
    units: str
    source: str
    # Shape coefficients when the sample was produced by morph(); None otherwise.
    betas: list[float] | None = None


def _meta_scalar(data: np.lib.npyio.NpzFile, key: str):
    arr = np.asarray(data[key])
    return arr.item() if arr.ndim == 0 else arr


def _meta_str(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    return str(_meta_scalar(data, key)) if key in data.files else default


def _meta_int(data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    return int(_meta_scalar(data, key)) if key in data.files else default


def load_mhr_npz(path: str | Path) -> MhrSample:
    """Load and validate an MHR ``.npz`` sample.

    Required arrays: ``verts`` (N, 3) float, ``faces`` (F, 3) int,
    ``joint_coords`` (127, 3), ``keypoints`` (70, 3). All face indices must lie in
    ``[0, N)``. Optional metadata: ``frame_index``, ``coordinate_frame``, ``units``,
    ``n_vertices``, ``source``. Raises :class:`ValueError` on any violation.
    """
    data = np.load(path, allow_pickle=False)

    for key in ("verts", "faces", "joint_coords", "keypoints"):
        if key not in data.files:
            raise ValueError(f"missing required key {key!r} in {path}")

    verts = np.asarray(data["verts"])
    faces = np.asarray(data["faces"])
    joint_coords = np.asarray(data["joint_coords"])
    keypoints = np.asarray(data["keypoints"])

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must have shape (N, 3), got {verts.shape}")
    if not np.issubdtype(verts.dtype, np.floating):
        raise ValueError(f"verts must be a float array, got dtype {verts.dtype}")
    n = verts.shape[0]

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"faces must be an integer array, got dtype {faces.dtype}")

    if joint_coords.shape != (N_JOINT_COORDS, 3):
        raise ValueError(
            f"joint_coords must have shape ({N_JOINT_COORDS}, 3), "
            f"got {joint_coords.shape}"
        )
    if keypoints.shape != (N_KEYPOINTS, 3):
        raise ValueError(
            f"keypoints must have shape ({N_KEYPOINTS}, 3), got {keypoints.shape}"
        )

    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= n):
        raise ValueError(
            f"face indices must be in [0, {n}), got "
            f"[{int(faces.min())}, {int(faces.max())}]"
        )

    if "n_vertices" in data.files:
        n_meta = _meta_int(data, "n_vertices", n)
        if n_meta != n:
            raise ValueError(
                f"n_vertices metadata ({n_meta}) does not match verts ({n})"
            )

    return MhrSample(
        verts=verts,
        faces=faces,
        joint_coords=joint_coords,
        keypoints=keypoints,
        frame_index=_meta_int(data, "frame_index", 0),
        coordinate_frame=_meta_str(data, "coordinate_frame", "unknown"),
        units=_meta_str(data, "units", "m"),
        source=_meta_str(data, "source", "unknown"),
    )

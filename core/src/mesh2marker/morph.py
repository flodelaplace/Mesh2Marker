"""Linear MHR shape morph in rest pose (pure numpy).

Pure core: stdlib + numpy only. No bpy, no pydantic, NO MHR / pymomentum dependency.

MHR shape is strictly linear in the shape coefficients (verified, ~3 micron error),
so a shape basis exported once from the upstream pipeline lets us regenerate the
rest-pose mesh, joints and keypoints exactly:

    V(b) = V0 + einsum("s,snc->nc", b, dV)   (and likewise for joints J and keypoints)

The basis (``shape_basis.npz``) carries V0/J0/KP0, the per-component displacements
dV/dJ/dKP, the faces, a reference ``delta`` and a small ``meta`` dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .mhr import MhrSample


@dataclass
class ShapeBasis:
    v0: np.ndarray  # (N, 3)
    j0: np.ndarray  # (J, 3)
    kp0: np.ndarray  # (K, 3)
    faces: np.ndarray  # (F, 3) int
    dv: np.ndarray  # (S, N, 3)
    dj: np.ndarray  # (S, J, 3)
    dkp: np.ndarray  # (S, K, 3)
    delta: float
    meta: dict

    @property
    def n_shape(self) -> int:
        return self.dv.shape[0]


def _meta_from_npz(data: np.lib.npyio.NpzFile) -> dict:
    if "meta" not in data.files:
        return {}
    raw = data["meta"]
    text = str(raw.item() if raw.ndim == 0 else raw)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_shape_basis(path) -> ShapeBasis:
    """Load and validate a ``shape_basis.npz``. Raises ValueError on shape mismatch."""
    data = np.load(path, allow_pickle=False)
    for key in ("V0", "J0", "KP0", "faces", "dV", "dJ", "dKP"):
        if key not in data.files:
            raise ValueError(f"missing required key {key!r} in {path}")

    v0 = np.asarray(data["V0"], dtype=float)
    j0 = np.asarray(data["J0"], dtype=float)
    kp0 = np.asarray(data["KP0"], dtype=float)
    faces = np.asarray(data["faces"])
    dv = np.asarray(data["dV"], dtype=float)
    dj = np.asarray(data["dJ"], dtype=float)
    dkp = np.asarray(data["dKP"], dtype=float)
    delta = float(data["delta"]) if "delta" in data.files else 1.0

    for name, arr in (("V0", v0), ("J0", j0), ("KP0", kp0)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")

    s = dv.shape[0]
    for name, disp, base in (("dV", dv, v0), ("dJ", dj, j0), ("dKP", dkp, kp0)):
        expected = (s, base.shape[0], 3)
        if disp.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {disp.shape}"
            )

    meta = _meta_from_npz(data)
    meta.setdefault("n_shape", int(s))
    return ShapeBasis(v0, j0, kp0, faces, dv, dj, dkp, delta, meta)


def _expand_betas(betas: np.ndarray, n_shape: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=float).reshape(-1)
    if betas.shape[0] > n_shape:
        raise ValueError(
            f"betas length {betas.shape[0]} exceeds n_shape {n_shape}"
        )
    if betas.shape[0] < n_shape:
        padded = np.zeros(n_shape)
        padded[: betas.shape[0]] = betas
        return padded
    return betas


def morph(basis: ShapeBasis, betas) -> MhrSample:
    """Regenerate the rest-pose sample for the given shape coefficients.

    ``betas`` may be shorter than ``n_shape`` (zero-padded) but not longer
    (ValueError). Returns an :class:`~mesh2marker.mhr.MhrSample` so the rest of the
    pipeline works unchanged on the morphed mesh.
    """
    coeffs = _expand_betas(betas, basis.n_shape)
    verts = basis.v0 + np.einsum("s,snc->nc", coeffs, basis.dv)
    joints = basis.j0 + np.einsum("s,snc->nc", coeffs, basis.dj)
    keypoints = basis.kp0 + np.einsum("s,snc->nc", coeffs, basis.dkp)
    return MhrSample(
        verts=verts,
        faces=basis.faces,
        joint_coords=joints,
        keypoints=keypoints,
        frame_index=0,
        coordinate_frame="mhr_rest",
        units=str(basis.meta.get("units", "meters")),
        source="morph",
        betas=[float(b) for b in coeffs],
    )


def component_displacements(basis: ShapeBasis) -> np.ndarray:
    """RMS vertex displacement of each shape component (length S).

    Ranks components by displacement AMPLITUDE, not by anatomical semantics.
    """
    return np.sqrt(np.mean(np.sum(basis.dv**2, axis=2), axis=1))

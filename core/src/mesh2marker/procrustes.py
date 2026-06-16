"""Umeyama (1991) similarity transform for pre-aligning corresponding clouds.

Pure core: stdlib + numpy only (numpy ships inside Blender's embedded Python, so
this respects the zero-wheel rule). No pydantic, no bpy.

Given corresponding source and target point clouds (e.g. MHR joint centres ->
OpenSim joint centres), :func:`procrustes_align` solves for the proper rotation
``R`` (det = +1), translation ``t`` and uniform scale ``s`` minimizing
``sum_i || s R source_i + t - target_i ||^2`` via the closed-form Umeyama SVD
solution, with reflection correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FrameAlignment


@dataclass
class SimilarityTransform:
    """A similarity transform ``y = s R x + t``.

    Stored as plain Python lists (JSON-serializable, consistent with the schema's
    :class:`~mesh2marker.models.FrameAlignment`); numpy is used only internally.
    """

    rotation: list[list[float]]  # 3x3
    translation: list[float]  # length 3
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to an ``Nx3`` array (or a single ``(3,)`` point)."""
        pts = np.asarray(points, dtype=float)
        rot = np.asarray(self.rotation, dtype=float)
        trans = np.asarray(self.translation, dtype=float)
        return self.scale * (pts @ rot.T) + trans


def procrustes_align(
    source: np.ndarray, target: np.ndarray, with_scale: bool = True
) -> SimilarityTransform:
    """Solve the Umeyama similarity transform mapping ``source`` onto ``target``.

    ``source`` and ``target`` must be ``Nx3`` arrays of the same shape, with
    ``N >= 3``. With ``with_scale=False`` the scale is forced to ``1`` (rigid
    alignment). Raises :class:`ValueError` on shape mismatch, too few points, or a
    degenerate (near-zero variance) source cloud.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)

    if src.shape != tgt.shape:
        raise ValueError(
            f"source and target must have the same shape, got {src.shape} "
            f"and {tgt.shape}"
        )
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"source and target must be Nx3 arrays, got {src.shape}")

    n = src.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 points, got {n}")

    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    var_src = float(np.sum(src_c**2) / n)
    if var_src < 1e-12:
        raise ValueError("degenerate source cloud: near-zero variance")

    # Cross-covariance of target and source (3x3), then SVD.
    cov = tgt_c.T @ src_c / n
    u, sing, vt = np.linalg.svd(cov)

    # Reflection correction: force a proper rotation (det = +1).
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1.0
    rot = u @ np.diag(signs) @ vt

    scale = float(np.sum(sing * signs) / var_src) if with_scale else 1.0
    trans = mu_tgt - scale * rot @ mu_src

    return SimilarityTransform(
        rotation=rot.tolist(),
        translation=trans.tolist(),
        scale=scale,
    )


def to_frame_alignment(st: SimilarityTransform) -> FrameAlignment:
    """Convert a :class:`SimilarityTransform` to a schema :class:`FrameAlignment`."""
    return FrameAlignment(
        rotation=[list(row) for row in st.rotation],
        translation=list(st.translation),
        scale=st.scale,
    )

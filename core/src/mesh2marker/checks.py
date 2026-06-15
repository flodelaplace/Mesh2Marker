"""Lightweight, pure-Python validation of a correspondence file.

No third-party dependency (no pydantic): this is meant to be callable from the
Blender add-on runtime. :func:`validate` returns a list of human-readable error
strings (empty list means valid). For strong, schema-level validation used by CI
and the Mesh2Sim pipeline, see :mod:`mesh2marker.validation`.
"""

from __future__ import annotations

from .models import SCHEMA_VERSION, CorrespondenceFile
from .vocabulary import LANDMARK_NAMES, SEGMENTS_BY_MODEL


def validate(corr: CorrespondenceFile) -> list[str]:
    """Return the list of validation errors for ``corr`` (empty if valid)."""
    errors: list[str] = []

    if corr.schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got {corr.schema_version!r}"
        )

    bodies = SEGMENTS_BY_MODEL.get(corr.opensim_model)
    if bodies is None:
        errors.append(f"unknown opensim model: {corr.opensim_model!r}")

    seen: set[str] = set()
    for m in corr.markers:
        if m.name in seen:
            errors.append(f"duplicate marker name: {m.name!r}")
        seen.add(m.name)

        if m.name not in LANDMARK_NAMES:
            errors.append(f"unknown marker name: {m.name!r}")

        if bodies is not None and m.opensim_body not in bodies:
            errors.append(
                f"unknown opensim body for model {corr.opensim_model!r}: "
                f"{m.opensim_body!r}"
            )

        if len(m.local_offset) != 3:
            errors.append(
                f"marker {m.name!r}: local_offset must have length 3, "
                f"got {len(m.local_offset)}"
            )

        if not m.mhr_vertices:
            errors.append(f"marker {m.name!r}: mhr_vertices must not be empty")
        else:
            for v in m.mhr_vertices:
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    errors.append(
                        f"marker {m.name!r}: mhr_vertices must be integers >= 0, "
                        f"got {v!r}"
                    )

    fa = corr.frame_alignment
    if len(fa.rotation) != 3 or any(len(row) != 3 for row in fa.rotation):
        errors.append("frame_alignment.rotation must be a 3x3 matrix")
    if len(fa.translation) != 3:
        errors.append("frame_alignment.translation must have length 3")

    return errors

"""JSON IO for the correspondence file (stdlib ``json`` only).

:func:`write` and :func:`read` are an exact round-trip: ``read(write(x)) == x``.
The on-disk layout follows the schema documented in CLAUDE.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import CorrespondenceFile, FrameAlignment, Marker


def write(corr: CorrespondenceFile, path: str | Path) -> None:
    """Serialize ``corr`` to ``path`` as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(corr), f, indent=2)
        f.write("\n")


def read(path: str | Path) -> CorrespondenceFile:
    """Read a correspondence file from ``path`` and rebuild the dataclasses."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _from_dict(data)


def _from_dict(data: dict) -> CorrespondenceFile:
    fa = data["frame_alignment"]
    frame_alignment = FrameAlignment(
        rotation=fa["rotation"],
        translation=fa["translation"],
        scale=fa["scale"],
    )
    markers = [
        Marker(
            name=m["name"],
            mhr_vertices=m["mhr_vertices"],
            opensim_body=m["opensim_body"],
            local_offset=m["local_offset"],
            fixed=m["fixed"],
            synthpose_index=m["synthpose_index"],
        )
        for m in data["markers"]
    ]
    return CorrespondenceFile(
        schema_version=data["schema_version"],
        mhr_topology_id=data["mhr_topology_id"],
        opensim_model=data["opensim_model"],
        marker_set=data["marker_set"],
        frame_alignment=frame_alignment,
        markers=markers,
    )

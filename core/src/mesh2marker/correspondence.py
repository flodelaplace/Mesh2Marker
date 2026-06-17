"""Bridge between session links and the correspondence-file schema (ticket 1).

Pure core: stdlib only (uses :mod:`mesh2marker.models` and :mod:`mesh2marker.io`).
No bpy, no pydantic, no new schema -- the existing ``CorrespondenceFile`` already
carries ``local_offset`` and ``fixed`` per marker, so nothing is extended here.

A "link" is the bpy-friendly session representation of one marker mapping: a dict
``{"marker", "vertex_indices", "opensim_body", "local_offset"?, "fixed"?,
"synthpose_index"?}``. These helpers convert links <-> the schema and reuse the
existing JSON IO.
"""

from __future__ import annotations

from .io import read, write
from .models import SCHEMA_VERSION, CorrespondenceFile, FrameAlignment, Marker


def _identity_frame_alignment() -> FrameAlignment:
    return FrameAlignment(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )


def links_to_correspondence(
    links: list[dict],
    *,
    mhr_topology_id: str,
    opensim_model: str,
    marker_set: str,
    frame_alignment: FrameAlignment | None = None,
) -> CorrespondenceFile:
    """Build a :class:`CorrespondenceFile` from session links + metadata."""
    if frame_alignment is None:
        frame_alignment = _identity_frame_alignment()
    markers = []
    for link in links:
        offset = link.get("local_offset") or [0.0, 0.0, 0.0]
        markers.append(
            Marker(
                name=link["marker"],
                mhr_vertices=[int(i) for i in link["vertex_indices"]],
                opensim_body=link.get("opensim_body", ""),
                local_offset=[float(x) for x in offset],
                fixed=bool(link.get("fixed", False)),
                synthpose_index=link.get("synthpose_index"),
            )
        )
    return CorrespondenceFile(
        schema_version=SCHEMA_VERSION,
        mhr_topology_id=mhr_topology_id,
        opensim_model=opensim_model,
        marker_set=marker_set,
        frame_alignment=frame_alignment,
        markers=markers,
    )


def correspondence_to_links(corr: CorrespondenceFile) -> list[dict]:
    """Flatten a :class:`CorrespondenceFile` back to session links (for reload)."""
    return [
        {
            "marker": m.name,
            "vertex_indices": list(m.mhr_vertices),
            "opensim_body": m.opensim_body,
            "local_offset": list(m.local_offset),
            "fixed": m.fixed,
            "synthpose_index": m.synthpose_index,
        }
        for m in corr.markers
    ]


def write_correspondence(
    links: list[dict],
    path,
    *,
    mhr_topology_id: str,
    opensim_model: str,
    marker_set: str,
    frame_alignment: FrameAlignment | None = None,
) -> CorrespondenceFile:
    """Build the schema object from links and write it as JSON (reusing io.write)."""
    corr = links_to_correspondence(
        links,
        mhr_topology_id=mhr_topology_id,
        opensim_model=opensim_model,
        marker_set=marker_set,
        frame_alignment=frame_alignment,
    )
    write(corr, path)
    return corr


def read_correspondence_links(path) -> tuple[CorrespondenceFile, list[dict]]:
    """Read a correspondence file and return ``(CorrespondenceFile, links)``."""
    corr = read(path)
    return corr, correspondence_to_links(corr)

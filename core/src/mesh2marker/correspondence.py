"""Bridge between session links and the correspondence-file schema (ticket 1).

Pure core: stdlib only at import time (uses :mod:`mesh2marker.models`,
:mod:`mesh2marker.io`, :mod:`mesh2marker.checks`). The export path additionally runs
the shared-contract validation (``mesh2sim.contracts.CorrespondenceMap``) WHEN that
package is importable -- a soft, lazy dependency, so the add-on still works (and stays
pydantic-free) without it. The on-disk schema matches the contract field-for-field;
identity values default to the canonical slug / topology id.

A "link" is the bpy-friendly session representation of one marker mapping: a dict
``{"marker", "vertex_indices", "opensim_body", "local_offset"?, "fixed"?,
"synthpose_index"?}``.
"""

from __future__ import annotations

from .checks import validate
from .io import read, write
from .models import SCHEMA_VERSION, CorrespondenceFile, FrameAlignment, Marker
from .vocabulary import MHR_TOPOLOGY_ID, MODEL_SLUG


def _identity_frame_alignment() -> FrameAlignment:
    return FrameAlignment(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )


def links_to_correspondence(
    links: list[dict],
    *,
    mhr_topology_id: str = MHR_TOPOLOGY_ID,
    opensim_model: str = MODEL_SLUG,
    marker_set: str = "mesh2marker",
    frame_alignment: FrameAlignment | None = None,
) -> CorrespondenceFile:
    """Build a :class:`CorrespondenceFile` from session links + metadata.

    Identity fields default to the canonical contract values (slug + ``mhr_v1``).
    """
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


def _to_contract_map(corr: CorrespondenceFile):
    """Build a mesh2sim.contracts.CorrespondenceMap from our CorrespondenceFile."""
    import mesh2sim.contracts as contracts

    fa = corr.frame_alignment
    return contracts.CorrespondenceMap(
        schema_version=corr.schema_version,
        mhr_topology_id=corr.mhr_topology_id,
        opensim_model=corr.opensim_model,
        marker_set=corr.marker_set,
        frame_alignment=contracts.FrameAlignment(
            rotation=fa.rotation, translation=fa.translation, scale=fa.scale
        ),
        markers=[
            contracts.CorrespondenceMarker(
                name=m.name,
                mhr_vertices=m.mhr_vertices,
                opensim_body=m.opensim_body,
                local_offset=m.local_offset,
                fixed=m.fixed,
                synthpose_index=m.synthpose_index,
            )
            for m in corr.markers
        ],
    )


def validate_for_export(corr: CorrespondenceFile) -> None:
    """Validate before writing. Raises :class:`ValueError` on any problem.

    Always runs the pure-Python vocabulary + structural checks (rejecting a marker
    name or segment outside the vocabulary). Additionally validates against
    ``mesh2sim.contracts.CorrespondenceMap`` when that package is importable (soft
    dependency); otherwise that step is skipped.
    """
    errors = validate(corr)
    if errors:
        raise ValueError("invalid correspondence file: " + "; ".join(errors))

    try:
        import mesh2sim.contracts  # noqa: F401  (presence check)
    except ImportError:
        return

    from pydantic import ValidationError

    try:
        _to_contract_map(corr)
    except ValidationError as exc:
        raise ValueError(
            f"CorrespondenceMap contract validation failed: {exc}"
        ) from exc


def write_correspondence(
    links: list[dict],
    path,
    *,
    mhr_topology_id: str = MHR_TOPOLOGY_ID,
    opensim_model: str = MODEL_SLUG,
    marker_set: str = "mesh2marker",
    frame_alignment: FrameAlignment | None = None,
) -> CorrespondenceFile:
    """Build the schema object from links, validate it, then write it as JSON."""
    corr = links_to_correspondence(
        links,
        mhr_topology_id=mhr_topology_id,
        opensim_model=opensim_model,
        marker_set=marker_set,
        frame_alignment=frame_alignment,
    )
    validate_for_export(corr)
    write(corr, path)
    return corr


def read_correspondence_links(path) -> tuple[CorrespondenceFile, list[dict]]:
    """Read a correspondence file and return ``(CorrespondenceFile, links)``."""
    corr = read(path)
    return corr, correspondence_to_links(corr)

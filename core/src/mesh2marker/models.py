"""Correspondence-file schema as pure dataclasses.

No third-party dependency: stdlib + typing only. This module must stay importable
inside Blender's embedded Python without pulling any compiled wheel, so it never
imports pydantic (that lives only in :mod:`mesh2marker.validation`).

Units are meters; local offsets are expressed in the segment frame
(right-handed, Y-up, OpenSim convention).
"""

from __future__ import annotations

from dataclasses import dataclass

# Schema version always written into the correspondence file. This is the shared
# Mesh2Sim contract wire version (mesh2sim.contracts.SCHEMA_VERSION), so exports load
# in the pipeline; bump together with the contract.
SCHEMA_VERSION = "0.1.0"


@dataclass
class FrameAlignment:
    """Rigid (+ uniform scale) alignment between the mesh and the OpenSim model.

    ``rotation`` is a 3x3 matrix (list of three rows of three floats),
    ``translation`` a length-3 vector, ``scale`` a single uniform factor.
    """

    rotation: list[list[float]]
    translation: list[float]
    scale: float


@dataclass
class Marker:
    """One anatomical marker mapped from MHR vertices onto an OpenSim body.

    ``mhr_vertices`` holds one or more MHR vertex indices (centroid if several).
    ``local_offset`` is a length-3 vector in the body's segment frame (meters).
    ``fixed`` is ``True`` for bony landmarks, ``False`` for soft tissue.
    ``synthpose_index`` is the optional SynthPose index (``None`` if not covered).
    """

    name: str
    mhr_vertices: list[int]
    opensim_body: str
    local_offset: list[float]
    fixed: bool
    synthpose_index: int | None = None


@dataclass
class CorrespondenceFile:
    """The reusable, shareable correspondence asset.

    Defined once on a template mesh (fixed MHR topology) and reused for every
    subject. The link to the Mesh2Sim pipeline is this file, not code.
    """

    schema_version: str
    mhr_topology_id: str
    opensim_model: str
    marker_set: str
    frame_alignment: FrameAlignment
    markers: list[Marker]

"""Lightweight pure-Python parser for OpenSim ``.osim`` models (4.x XML).

Uses only :mod:`xml.etree.ElementTree` from the stdlib: no ``opensim`` package,
no pydantic, no compiled wheel. This keeps the parser importable inside Blender's
embedded Python.

Scoping discipline (load-bearing): every extraction is strictly scoped to its set
(``BodySet`` / ``JointSet`` / ``MarkerSet``). We NEVER do a global search for a tag
such as ``socket_parent_frame`` — it occurs ~1000 times across joints, wrap objects
and muscle path points. ``ForceSet`` (muscles), ``ConstraintSet``,
``ContactGeometrySet``, wrap objects, controllers, probes and ``FrameGeometry`` are
ignored entirely.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# --- helpers ---------------------------------------------------------------


def _floats(text: str | None) -> list[float]:
    """Parse whitespace-separated floats; empty list for missing text."""
    if not text:
        return []
    return [float(tok) for tok in text.split()]


def _last_segment(ref: str | None) -> str:
    """Trailing segment of a connectee path or bare name.

    ``"/jointset/hip_r/pelvis_offset"`` -> ``"pelvis_offset"``,
    ``"pelvis_offset"`` -> ``"pelvis_offset"``.
    """
    if ref is None:
        return ""
    ref = ref.strip()
    return ref.rsplit("/", 1)[-1] if "/" in ref else ref


def _strip_body_path(ref: str | None) -> str:
    """Resolve a connectee path to a body name when recognizable.

    Strips the ``/bodyset/`` prefix and maps ``/ground`` to ``"ground"``. If the
    path is not a recognizable direct body reference, the stripped raw value is
    returned unchanged rather than raising.
    """
    if ref is None:
        return ""
    ref = ref.strip()
    if ref.startswith("/bodyset/"):
        return ref[len("/bodyset/") :]
    if ref == "/ground" or ref.endswith("/ground"):
        return "ground"
    return ref


# --- data model ------------------------------------------------------------


@dataclass
class OsimGeometry:
    mesh_name: str
    mesh_file: str
    scale_factors: list[float]


@dataclass
class OsimBody:
    name: str
    geometries: list[OsimGeometry]


@dataclass
class OsimFrameOffset:
    translation: list[float]
    orientation: list[float]


@dataclass
class OsimJoint:
    name: str
    joint_type: str
    parent_body: str
    child_body: str
    parent_offset: OsimFrameOffset
    child_offset: OsimFrameOffset
    coordinates: list[str]


@dataclass
class OsimMarker:
    name: str
    parent_body: str
    location: list[float]


@dataclass
class OsimModel:
    name: str
    bodies: list[OsimBody]
    joints: list[OsimJoint]
    markers: list[OsimMarker]

    def adjacency(self) -> dict[str, list[str]]:
        """Parent-body -> child-bodies adjacency (the kinematic tree)."""
        tree: dict[str, list[str]] = {}
        for joint in self.joints:
            tree.setdefault(joint.parent_body, []).append(joint.child_body)
        return tree


# --- per-set extraction ----------------------------------------------------


def _parse_bodies(model_el: ET.Element) -> list[OsimBody]:
    bodies: list[OsimBody] = []
    objects = model_el.find("BodySet/objects")
    if objects is None:
        return bodies
    for body_el in objects.findall("Body"):
        geometries: list[OsimGeometry] = []
        # Only attached_geometry meshes; FrameGeometry is intentionally ignored.
        attached = body_el.find("attached_geometry")
        if attached is not None:
            for mesh_el in attached.findall("Mesh"):
                geometries.append(
                    OsimGeometry(
                        mesh_name=mesh_el.get("name", ""),
                        mesh_file=(mesh_el.findtext("mesh_file") or "").strip(),
                        scale_factors=_floats(mesh_el.findtext("scale_factors")),
                    )
                )
        bodies.append(OsimBody(name=body_el.get("name", ""), geometries=geometries))
    return bodies


def _parse_offset_frames(
    joint_el: ET.Element,
) -> dict[str, tuple[str, OsimFrameOffset]]:
    """Map each PhysicalOffsetFrame name -> (resolved body, offset).

    Scoped to the joint's own ``<frames>`` block, so the ``socket_parent`` reads
    here cannot leak into other components.
    """
    frames: dict[str, tuple[str, OsimFrameOffset]] = {}
    frames_el = joint_el.find("frames")
    if frames_el is None:
        return frames
    for pof in frames_el.findall("PhysicalOffsetFrame"):
        body = _strip_body_path(pof.findtext("socket_parent"))
        offset = OsimFrameOffset(
            translation=_floats(pof.findtext("translation")),
            orientation=_floats(pof.findtext("orientation")),
        )
        frames[pof.get("name", "")] = (body, offset)
    return frames


def _resolve_frame(
    frame_ref: str | None, frames: dict[str, tuple[str, OsimFrameOffset]]
) -> tuple[str, OsimFrameOffset]:
    """Resolve a joint socket frame reference to (body, offset).

    Handles both a bare offset-frame name and a path (last segment). If the ref
    does not match a PhysicalOffsetFrame, it is treated as a direct body
    reference with a zero offset.
    """
    key = _last_segment(frame_ref)
    if key in frames:
        return frames[key]
    return _strip_body_path(frame_ref), OsimFrameOffset([], [])


def _parse_joints(model_el: ET.Element) -> list[OsimJoint]:
    joints: list[OsimJoint] = []
    objects = model_el.find("JointSet/objects")
    if objects is None:
        return joints
    # Every child is a joint, whatever its concrete type (handled generically).
    for joint_el in list(objects):
        frames = _parse_offset_frames(joint_el)
        parent_body, parent_offset = _resolve_frame(
            joint_el.findtext("socket_parent_frame"), frames
        )
        child_body, child_offset = _resolve_frame(
            joint_el.findtext("socket_child_frame"), frames
        )
        coordinates: list[str] = []
        coords_el = joint_el.find("coordinates")
        if coords_el is not None:
            coordinates = [c.get("name", "") for c in coords_el.findall("Coordinate")]
        joints.append(
            OsimJoint(
                name=joint_el.get("name", ""),
                joint_type=joint_el.tag,
                parent_body=parent_body,
                child_body=child_body,
                parent_offset=parent_offset,
                child_offset=child_offset,
                coordinates=coordinates,
            )
        )
    return joints


def _parse_markers(model_el: ET.Element) -> list[OsimMarker]:
    markers: list[OsimMarker] = []
    objects = model_el.find("MarkerSet/objects")
    if objects is None:
        return markers
    for marker_el in objects.findall("Marker"):
        markers.append(
            OsimMarker(
                name=marker_el.get("name", ""),
                parent_body=_strip_body_path(
                    marker_el.findtext("socket_parent_frame")
                ),
                location=_floats(marker_el.findtext("location")),
            )
        )
    return markers


# --- entry point -----------------------------------------------------------


def parse_osim(path: str | Path) -> OsimModel:
    """Parse an OpenSim ``.osim`` file into an :class:`OsimModel`."""
    root = ET.parse(path).getroot()
    model_el = root.find("Model")
    if model_el is None:
        raise ValueError(f"no <Model> element found in {path}")
    return OsimModel(
        name=model_el.get("name", ""),
        bodies=_parse_bodies(model_el),
        joints=_parse_joints(model_el),
        markers=_parse_markers(model_el),
    )

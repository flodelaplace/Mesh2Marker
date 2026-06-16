#!/usr/bin/env python3
"""Generate the frozen reference snapshot of the production OpenSim model.

Usage::

    python scripts/extract_reference_model.py <path-to-model.osim>

Parses the given ``.osim`` with :mod:`xml.etree.ElementTree` and emits
``reference/opensim_model.json`` with VERBATIM strings (no case normalization;
the marker ``location`` is kept as the raw string from the file, not reparsed to
floats). Extraction is strictly scoped per set (BodySet / JointSet / MarkerSet);
muscles, constraints, wrap objects, contact geometry and FrameGeometry are
ignored. As a consistency cross-check it also runs ``mesh2marker.osim.parse_osim``
and asserts the body and marker counts agree (dogfooding the parser).

The model path is taken from the command line; it is never hardcoded (the real
model lives under the gitignored ``local_models/``).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from mesh2marker.osim import parse_osim

# Output lives at the repo root, regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reference" / "opensim_model.json"

SLUG = "Pose2Sim_Wholebody"


def _strip_bodyset(ref: str | None) -> str:
    if ref is None:
        return ""
    ref = ref.strip()
    return ref[len("/bodyset/") :] if ref.startswith("/bodyset/") else ref


def _extract_bodies(model_el: ET.Element) -> list[str]:
    objects = model_el.find("BodySet/objects")
    if objects is None:
        return []
    return [b.get("name", "") for b in objects.findall("Body")]


def _extract_coordinates(model_el: ET.Element) -> list[dict]:
    coordinates: list[dict] = []
    objects = model_el.find("JointSet/objects")
    if objects is None:
        return coordinates
    for joint_el in list(objects):
        coords_el = joint_el.find("coordinates")
        if coords_el is None:
            continue
        for c in coords_el.findall("Coordinate"):
            coordinates.append(
                {
                    "name": c.get("name", ""),
                    "joint": joint_el.get("name", ""),
                    "joint_type": joint_el.tag,
                }
            )
    return coordinates


def _extract_markers(model_el: ET.Element) -> list[dict]:
    markers: list[dict] = []
    objects = model_el.find("MarkerSet/objects")
    if objects is None:
        return markers
    for mk in objects.findall("Marker"):
        markers.append(
            {
                "name": mk.get("name", ""),
                "parent_body": _strip_bodyset(mk.findtext("socket_parent_frame")),
                "location": mk.findtext("location"),
            }
        )
    return markers


def build_reference(path: Path) -> dict:
    model_el = ET.parse(path).getroot()
    version = model_el.get("Version")
    model_el = model_el.find("Model")
    if model_el is None:
        raise ValueError(f"no <Model> element found in {path}")

    bodies = _extract_bodies(model_el)
    markers = _extract_markers(model_el)

    # Cross-check against the production parser (dogfood the per-set scoping).
    parsed = parse_osim(path)
    assert len(parsed.bodies) == len(bodies), (
        f"body count mismatch: parser={len(parsed.bodies)} extractor={len(bodies)}"
    )
    assert len(parsed.markers) == len(markers), (
        f"marker count mismatch: parser={len(parsed.markers)} "
        f"extractor={len(markers)}"
    )

    return {
        "schema_note": (
            "Frozen reference snapshot of the production OpenSim model. Verbatim "
            "strings, no case normalization. Shared contract with the Mesh2Sim "
            "pipeline."
        ),
        "model_identity": {
            "slug": SLUG,
            "model_name_verbatim": model_el.get("name"),
            "opensim_document_version": version,
            "derived_family": (
                "Full-body Pose2Sim model (Pagnon et al. 2021), adapted from "
                "Beaucage-Gauvreau et al. with Rajagopal et al. knee angles. Not "
                "stock Rajagopal2016, not Lai-Uhlrich."
            ),
            "credits_verbatim": model_el.findtext("credits"),
            "publications_verbatim": model_el.findtext("publications"),
        },
        "units_and_axes": {
            "length_units": model_el.findtext("length_units"),
            "force_units": model_el.findtext("force_units"),
            "gravity": model_el.findtext("gravity"),
            "axis_convention": (
                "OpenSim standard: right-handed, Y-up (gravity along -Y confirms "
                "vertical = Y)"
            ),
        },
        "bodies": bodies,
        "coordinates": _extract_coordinates(model_el),
        "markerset": {
            "status": (
                "fully populated; all markers attached directly to a body; all "
                "parent_body present in bodies"
            ),
            "count": len(markers),
            "markers": markers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osim_path", type=Path, help="Path to the .osim model file")
    args = parser.parse_args()

    reference = build_reference(args.osim_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: "
        f"{len(reference['bodies'])} bodies, "
        f"{reference['markerset']['count']} markers, "
        f"{len(reference['coordinates'])} coordinates"
    )


if __name__ == "__main__":
    main()

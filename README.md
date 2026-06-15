# Mesh2Marker

GUI tool (Blender add-on) that displays an MHR parametric body mesh and an
OpenSim musculoskeletal model in the same 3D scene, lets the user map mesh
vertices to anatomical markers on OpenSim segments, and exports a reusable
correspondence file.

Mesh2Marker is an annex open-source project of the Mesh2Sim markerless-
biomechanics pipeline. The link to the pipeline is the **correspondence file**,
not code.

## Why this tool

MHR topology is fixed: a given vertex index is the same anatomical point across
all subjects, regardless of morphology. The mapping is defined **once** on a
template mesh and reused for every subject. Mesh2Marker therefore produces a
reusable, shareable asset (the equivalent of a standard marker set in optical
mocap), not a per-subject output.

## Repository layout: two artifacts, two channels, two licenses

This repository ships two distinct artifacts.

### `core/` — the business logic (PyPI, MIT)

Pure Python, **no `bpy`**. Parsing of `.osim`, Procrustes alignment, local-offset
computation, correspondence-file schema and IO, validation against the shared
vocabulary. Fully testable with plain pytest.

- **Channel:** published to PyPI — `pip install mesh2marker`.
- **License:** MIT.
- **Audience:** developers and the Mesh2Sim pipeline.

### `blender_addon/` — the Blender extension (Blender platform, GPL-3.0-or-later)

The thin `bpy` layer: UI, vertex picking, scene display, panels. It calls
`core/`; no business logic lives here.

- **Channel:** distributed as a Blender extension (4.2+), via "Install from Disk"
  `.zip` or the Blender Extensions Platform. **Never** via pip for end users.
- **License:** GPL-3.0-or-later (any code importing `bpy` is a derivative work of
  Blender). An MIT `core` consumed by the GPL add-on is fine; the dependency
  direction is `core` → add-on, never the reverse.

## Status

Early scaffolding. Business logic (`.osim` parsing, Procrustes alignment,
correspondence-file schema) is not implemented yet.

## Development

```bash
# Core: WSL / Linux, plain pytest
cd core
pytest
```

The Blender add-on is built with `blender --command extension build`.

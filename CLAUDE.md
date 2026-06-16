# Mesh2Marker

GUI tool (Blender add-on) that displays an MHR parametric body mesh and an OpenSim
musculoskeletal model in the same 3D scene, lets the user map mesh vertices to anatomical
markers on OpenSim segments, and exports a reusable correspondence file. Annex open-source
project of the Mesh2Sim markerless-biomechanics pipeline. The link to the pipeline is the
correspondence file, not code.

## Central insight
MHR topology is fixed: a given vertex index is the same anatomical point across all subjects,
regardless of morphology. The mapping is defined ONCE on a template mesh and reused for every
subject. This tool produces a reusable, shareable asset (the equivalent of a standard marker
set in optical mocap), not a per-subject output.

## Architecture: strict two-layer split
- `core/` : pure Python, NO bpy. Business logic, fully testable with plain pytest.
  Parsing of .osim, Procrustes alignment, local offset computation, correspondence-file
  schema and IO, validation against the shared vocabulary.
- `blender_addon/` : the bpy layer (UI, vertex picking, scene display, panels). Thin.
  It calls `core/`. No business logic lives here.

Dependency direction is one-way: `blender_addon` depends on `core`. `core` NEVER imports bpy
and NEVER depends on the add-on.

## Distribution: two artifacts, two channels, two licenses
- `core` -> published to PyPI, `pip install mesh2marker`. License MIT. For developers and for
  the Mesh2Sim pipeline.
- `blender_addon` -> distributed as a Blender extension (4.2+), via "Install from Disk" .zip
  or the Blender Extensions Platform. NEVER pip for end users. License GPL-3.0-or-later
  (any code importing bpy is a derivative work of Blender). MIT core consumed by GPL add-on
  is fine; the dependency goes core -> add-on, not the reverse.

## Dependencies and packaging (load-bearing rule)
The add-on runtime must contain ZERO compiled third-party wheels, so it installs cleanly on
every platform.
- numpy: already bundled inside Blender's embedded Python. Use it freely (Procrustes SVD).
- stdlib json for file IO.
- pydantic: NEVER imported in the add-on runtime path. It lives only in `core.validation`
  (an optional `[validation]` extra), used by CI and by the Mesh2Sim pipeline, which revalidate
  the file anyway. The add-on uses dataclasses + light checks.
- The only wheel ever bundled in a release is our own `core`, built as a pure-python
  (py3-none-any) wheel. In the built .zip it is listed in the manifest `wheels = [...]`.
- Dev loop: do NOT rely on bundled wheels (Blender does not install wheels from a local
  repository, only from a built .zip or the platform). During development the add-on adds the
  core source dir to sys.path so edits are reflected live.

## Python version
Target Python 3.11 (Blender 4.x embedded interpreter). The core must stay 3.11-compatible so it
runs both inside Blender and in Linux CI.

## Integration with Mesh2Sim
No code dependency. The link is the correspondence file (JSON). Three alignment points with the
pipeline's shared contract (`mesh2sim.contracts`):

- Stable model slug = `"Pose2Sim_Wholebody"`. The verbatim model name
  `"Pose2Sim_WithMusclesAndConstraints"` is kept as metadata only. The frozen snapshot of the
  production model is `reference/opensim_model.json` (regenerable via
  `scripts/extract_reference_model.py`).
- The 73 marker names are the shared landmark vocabulary; correspondences MUST use these exact
  names, case-sensitive. Gotchas: `"Abdomen"` is capitalized, `"RWrist_hand"` / `"LWrist_hand"`
  use an underscore, `"RFAradius"` / `"RFAulna"` are both on body `radius_r` (mirror on
  `radius_l`).
- The exported correspondence file MUST conform to the `CorrespondenceMap` schema defined in the
  pipeline's `mesh2sim.contracts` (`schema_version`, `mhr_topology_id`, `opensim_model`,
  `marker_set`, `frame_alignment`{`rotation`, `translation`, `scale`}, `markers`[`name`,
  `mhr_vertices`, `opensim_body`, `local_offset`, `fixed`, `synthpose_index`]). When
  `mesh2sim.contracts` is available it will be installed in Blender's Python to validate exports
  against the shared schema.

## Conventions
Meters. Right-handed Y-up frame (OpenSim convention). Local offsets expressed in the segment
frame. `schema_version` always written in the correspondence file.

## Correspondence file schema (target)
JSON with: schema_version, mhr_topology_id, opensim_model, marker_set, frame_alignment
(rotation/translation/scale), and a markers list. Each marker: name (shared vocabulary),
mhr_vertices (one or more indices, centroid if several), opensim_body, local_offset (meters,
segment frame), fixed (true = bony, false = soft-tissue), synthpose_index (optional, null if
not covered).

## Guardrails
- No business logic in the bpy layer. Everything testable goes in `core/`.
- Parse .osim as lightweight standalone XML. Do NOT embed the heavy `opensim` package.
- Small, atomic commits. Tests first on the core.
- All repo content (code, comments, commit messages, docs) in English.

## Dev and test
- core: WSL, plain pytest, Linux CI.
- blender_addon: tested via Blender background mode (blender --background --python ...) or in a
  native Blender; the precise interactive dev loop is set up when the Blender tickets start.
- Build the extension with: blender --command extension build
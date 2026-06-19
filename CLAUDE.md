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

## Role of Mesh2Marker and bridge stage order
Durable reference (we clarified this; do not lose it). Mesh2Marker is a pip-installable
CONFIGURATION tool, NOT a processing engine. It never touches video and never runs
frame-by-frame. It acts at two moments, both OUTSIDE the processing loop:

1. Global configuration, once per model: manual placement of markers on the MHR mesh,
   producing the CorrespondenceMap (markers -> invariant mhr_vertices). A default model and its
   map will ship in the Mesh2Marker repo so this step need not be redone.
2. Per-subject personalization, once per participant, AFTER inference: from the subject mesh's
   morphology parameters (produced by inference) and the CorrespondenceMap, Mesh2Marker
   generates a personalized .osim with the reference markers placed on the subject's skin. This
   step CANNOT precede inference, since it consumes the shape parameters inference produces.
   Novel contribution: a reference marker set adapted to individual morphology, not to an
   average subject.

Frame-by-frame processing (extracting the observed marker positions on the subject's meshes via
the mhr_vertices, centroid if several) is done by the PIPELINE, not by Mesh2Marker. It is a
simple computation (read vertices, average) that belongs to the BodyEstimate ->
AnatomicalObservation adapter.

Consistency via shape-lock: the subject's shape is locked for the trial, so the personalized
.osim (step 2) and the frame-by-frame processed meshes start from the same morphology, which
aligns the reference model with the observations.

Bridge stage order: frontend (inference, per-frame BodyEstimate) -> shape-lock -> per-subject
personalized .osim generation via Mesh2Marker (step 2) -> frame-by-frame adapter (observed
positions -> AnatomicalObservation) -> fusion/assembly and frame transform -> OpenSim Scale + IK
on the personalized .osim.

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

## Vision and future direction
Forward-looking context for future sessions. These are VISION decisions, not tickets to
implement now; they explain where the tool is heading and why.

### Bidirectional tool (an optimiser in both directions)
Mesh2Marker is bidirectional — think of it as an optimiser that runs both ways:
- Direction 1 — picking (ticket 5b): for each OpenSim marker, find the index of the
  corresponding MHR vertex. The vertex index is the reference datum, fixed (MHR topology is
  fixed), and exact regardless of alignment quality.
- Direction 2 — repositioning (ticket 5c): the inverse — move the OpenSim marker positions so
  they sit on the SKIN at the chosen vertex (far enough out from the bone). It is the markers
  that move, never the vertices; a marker's position is DERIVED from its vertex index.

Biomechanical rationale: the upstream pipeline's virtual markers also come from the skin (the
mesh), so moving the model's markers onto the skin makes the marker set consistent with the data
(skin against skin), removing the systematic bone/skin offset the IK currently sees.

### Morphology plan (future extension of 5c — do NOT implement now)
- Because the map is indexed by vertex (not by position), changing the MHR mesh shape
  automatically changes each vertex's skin position, hence the marker positions. A heavier
  subject => skin farther from the bone => markers farther out, DETERMINISTICALLY (no free
  parameter to optimise; the subject's skin gives the answer).
- Two morphology sources, same machinery: (a) manual morphology via a slider to build archetypes
  (thin/average/heavy); (b) estimated morphology of the real subject (their betas computed by the
  upstream pipeline) for a person-specific model.
- 5c will be generalised: take an MHR mesh of ANY morphology (.npz) as input, read the vertices
  at the map's indices, express them in the OpenSim segment frame (via the subject's MHR joints,
  not via the global pre-alignment, for robustness), and write a .osim with the markers at those
  skin positions.
- Architecture constraint kept: Mesh2Marker does NOT depend on MHR / pymomentum (zero compiled
  wheels). The morphology slider will be PURE NUMPY: linear shape deformation in rest pose
  (template + sum(beta_i * direction_i)). The shape basis (template + 45 identity directions +
  shape->joints regression) will be exported ONCE from the upstream pipeline (which has MHR) to a
  simple .npz that Mesh2Marker consumes. To confirm: linearity of MHR shape at rest.

### Shape basis: 45-identity and the extended 73 (identity + scale) — DELIVERED
The upstream pipeline now exports an EXTENDED shape basis and Mesh2Marker consumes it:
`local_models/mhr_shape_basis_extended.npz` (gitignored). Layout of the concatenated shape
vector, length **73 = 45 identity + 28 scale**:
- indices `[0:45]`: the original identity directions, **bit-for-bit identical** to the legacy
  `local_models/mhr_shape_basis.npz` (verified). `V0/J0/KP0/faces` are inherited verbatim, so
  markers already picked on `V0` stay semantically aligned across both bases.
- indices `[45:73]`: 28 scale directions from the MHR rig. The effective scale dimension is 24
  (rows `[45,46,47,61]` are identically zero — dead PCA components, kept at 28 for contract
  consistency). The scale block of `dKP` is zero-padded: this basis does NOT regenerate the 70
  keypoints for scale-only changes — fine for marker placement (which reads vertices), but any
  downstream stage needing scale-dependent keypoints must recompute them.

Consequence for the skeleton fit: `segment_align.compute_segment_transforms` scales each long
bone (femur/tibia/humerus/forearm, L+R) from the **127-joint MHR rig** (`sample.joint_coords`,
which carries `dJ[45:73]`), NOT from the 70 keypoints (`dKP[45:73]==0`). With keypoints the long
bones stayed at identity scale for any scale-only morphology; with rig joints they now scale with
the subject's full morphology. The joint indices (hip/knee/ankle/shoulder/elbow/wrist, L+R) come
from the pipeline. Pelvis and torso keep their vertex/keypoint landmark fit (already scale-aware
via `dV`); head and hands stay on keypoints (no rig joint in the provided map, non-critical).

C2 (per-subject .osim generation) now consumes a single concatenated 73-vector
`[shape_params(45), scale_params(28)]`, passed to `morph(basis, betas)` / the headless
`mesh2marker generate --basis … --betas …` CLI. `morph` is component-agnostic: it reads
`n_shape` from the basis (`dV.shape[0]`), accepts the full 73-vector, zero-pads shorter ones
(identity-only), and rejects longer ones. The interactive Blender sliders still drive only the
identity block (12 of 32 exposed); the full 73-vector is the headless C2 path. The legacy
45-basis remains loadable for backward compatibility.
# Reference snapshot

`opensim_model.json` is the **frozen snapshot of the production OpenSim model**
(`Pose2Sim_Wholebody`) and the **shared contract** with the Mesh2Sim pipeline.
It is committed on purpose: it is the single source of truth for the marker and
segment vocabulary, and it is what `core/tests/test_contract_sync.py` checks the
code vocabulary (`mesh2marker.vocabulary`) against.

## What it contains

Verbatim strings extracted from the `.osim` (no case normalization; marker
`location` is kept as the raw string from the file, not reparsed to floats):

- `model_identity` — stable slug, verbatim model name, document version, derived
  family, and verbatim credits / publications.
- `units_and_axes` — length / force units, gravity, axis convention.
- `bodies` — the 30 body (segment) names, in file order.
- `coordinates` — every joint coordinate with its joint and joint type.
- `markerset` — the 73 markers, each with `name`, `parent_body` and verbatim
  `location`.

## Regenerating

The snapshot is reproducible. The production model lives under the gitignored
`local_models/` (never committed); pass its path explicitly:

```bash
python scripts/extract_reference_model.py local_models/Model_Flodelaplace_mocap.osim
```

The generator parses the model with the stdlib XML reader, scopes extraction
strictly per set (`BodySet` / `JointSet` / `MarkerSet`), and cross-checks the body
and marker counts against `mesh2marker.osim.parse_osim` before writing the file.

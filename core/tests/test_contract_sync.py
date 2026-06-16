"""Lock the code vocabulary and the frozen reference contract together.

This is a hard test (not skipped): ``reference/opensim_model.json`` is committed,
so it is present in CI. It guarantees that ``vocabulary.py`` cannot drift away
from the frozen production-model snapshot without a test failure.
"""

import json
from pathlib import Path

from mesh2marker.vocabulary import LANDMARK_NAMES, SEGMENTS_BY_MODEL

# core/tests/ -> core/ -> repo root.
REFERENCE = (
    Path(__file__).resolve().parents[2] / "reference" / "opensim_model.json"
)


def _load() -> dict:
    with open(REFERENCE, encoding="utf-8") as f:
        return json.load(f)


def test_markers_match_landmark_vocabulary():
    data = _load()
    names = {m["name"] for m in data["markerset"]["markers"]}
    assert names == LANDMARK_NAMES


def test_bodies_match_segment_vocabulary():
    data = _load()
    assert set(data["bodies"]) == SEGMENTS_BY_MODEL["Pose2Sim_Wholebody"]


def test_slug():
    data = _load()
    assert data["model_identity"]["slug"] == "Pose2Sim_Wholebody"

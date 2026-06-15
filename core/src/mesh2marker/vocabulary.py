"""Shared marker / segment vocabulary — the single alignment point.

This module is the ONE place where the correspondence file's vocabulary is
defined, and it must stay aligned with the Mesh2Sim shared contract
(``mesh2sim.contracts``), itself based on the standard mocap marker set
(OpenCap 43-marker reference) and the per-model OpenSim segment (body) names.

IMPORTANT: the lists below are deliberately REDUCED seeds, not the full sets.
They are real and safe values, but the complete 43-marker set and the complete
per-model body lists are still to be finalized against ``mesh2sim.contracts``.
When that contract is reachable, reconcile here (and nowhere else).
"""

from __future__ import annotations

# Marker names (subset of the OpenCap reference set). Seed — to be completed.
LANDMARK_NAMES: frozenset[str] = frozenset(
    {
        "r_asis",
        "l_asis",
        "r_psis",
        "l_psis",
        "r_knee",
        "l_knee",
        "r_ankle",
        "l_ankle",
    }
)

# OpenSim body names per model. Seed — to be completed per model.
SEGMENTS_BY_MODEL: dict[str, frozenset[str]] = {
    "Rajagopal2016": frozenset(
        {
            "pelvis",
            "femur_r",
            "femur_l",
            "tibia_r",
            "tibia_l",
            "calcn_r",
            "calcn_l",
            "torso",
        }
    ),
}

"""Shared marker / segment vocabulary — the single alignment point.

This module is the ONE place where the correspondence file's vocabulary is
defined, and it must stay aligned with the Mesh2Sim shared contract
(``mesh2sim.contracts``). The values below are the FULL set for the production
model ``Pose2Sim_Wholebody`` (frozen snapshot in ``reference/opensim_model.json``):
the 73 marker names are the shared landmark vocabulary, and the 30 body names are
the model's segments.

Names are case-sensitive and must be used verbatim. Gotchas worth remembering:
``"Abdomen"`` is capitalized; ``"RWrist_hand"`` / ``"LWrist_hand"`` use an
underscore; ``"RFAradius"`` / ``"RFAulna"`` both sit on body ``radius_r`` (mirror
``LFAradius`` / ``LFAulna`` on ``radius_l``).
"""

from __future__ import annotations

# The 73 marker names of Pose2Sim_Wholebody — the shared landmark vocabulary.
LANDMARK_NAMES: frozenset[str] = frozenset(
    {
        # Head
        "Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "HTOP",
        # Spine / neck virtual chain
        "c_spine0",
        "c_spine1",
        "c_spine2",
        "c_spine3",
        "c_neck",
        "c_head",
        # Shoulders / torso
        "RACR",
        "LACR",
        "C7",
        "RCLAV",
        "LCLAV",
        # Arms
        "RLEL",
        "RMEL",
        "RFAradius",
        "RFAulna",
        "LLEL",
        "LMEL",
        "LFAradius",
        "LFAulna",
        # Pelvis
        "RASI",
        "LASI",
        "RPSI",
        "LPSI",
        # Legs
        "RLFC",
        "RMFC",
        "RLMAL",
        "RMMAL",
        "RCAL",
        "RTOE",
        "RMT5",
        "LLFC",
        "LMFC",
        "LLMAL",
        "LMMAL",
        "LCAL",
        "LTOE",
        "LMT5",
        # Hands
        "RWrist_hand",
        "RThumb",
        "RIndex",
        "RPinky",
        "RIndexTip",
        "RPinkyTip",
        "LWrist_hand",
        "LThumb",
        "LIndex",
        "LPinky",
        "LIndexTip",
        "LPinkyTip",
        # Upper-arm marker clusters
        "RHTO",
        "RHAP",
        "RHBA",
        "RHFR",
        "RFRM",
        "LHTO",
        "LHAP",
        "LHBA",
        "LHFR",
        "LFRM",
        # Thigh / shank marker clusters
        "RFLT",
        "RFLB",
        "RSHN",
        "RTIB",
        "LFLT",
        "LFLB",
        "LSHN",
        "LTIB",
    }
)

# OpenSim body (segment) names per model. Pose2Sim_Wholebody has 30 bodies.
SEGMENTS_BY_MODEL: dict[str, frozenset[str]] = {
    "Pose2Sim_Wholebody": frozenset(
        {
            "pelvis",
            "sacrum",
            "femur_r",
            "patella_r",
            "tibia_r",
            "talus_r",
            "calcn_r",
            "toes_r",
            "femur_l",
            "patella_l",
            "tibia_l",
            "talus_l",
            "calcn_l",
            "toes_l",
            "lumbar5",
            "lumbar4",
            "lumbar3",
            "lumbar2",
            "lumbar1",
            "torso",
            "head",
            "Abdomen",
            "humerus_r",
            "ulna_r",
            "radius_r",
            "hand_r",
            "humerus_l",
            "ulna_l",
            "radius_l",
            "hand_l",
        }
    ),
}

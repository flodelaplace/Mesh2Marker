"""Per-body skeletal scale factors from the MHR rig joints (pure numpy).

The subject's bone dimensions are read from the MHR **rig joints**
(``sample.joint_coords``). Those joints respond ONLY to the scale block of the shape
basis: the 45 identity/soft-tissue directions move them by ~0 (measured), so a heavier
subject does NOT inflate the skeleton -- the scale factors here are morphology/fat-
immune by construction. Bone LENGTH comes from the skeleton; corpulence only pushes skin
markers outward (handled separately by the marker repositioning).

These factors are meant to be handed to OpenSim's ScaleTool (built into a ScaleSet)
downstream. Mesh2Marker never scales the ``.osim`` itself (lightweight XML only, no
``opensim`` package): scaling bodies + inertials + geometry + muscle points + wrap
objects + coupled coordinates correctly is exactly the ScaleTool's job.

Each factor is isotropic ``(s, s, s)`` and RELATIVE to the template morph (betas = 0),
so betas = 0 gives ``1.0`` everywhere regardless of the OpenSim model's base sizes.
"""

from __future__ import annotations

import numpy as np

from .morph import ShapeBasis, morph

# OpenSim body -> (proximal, distal) index into the MHR 127-joint rig. Segment length
# is the distance between the two joints. Indices validated against the rig geometry
# and the mhr70 joint keypoints:
#   - ankle = 20 (r) / 4 (l): the mhr70 ankle keypoint sits 0.15-0.18 cm from these,
#     vs ~0.5 cm from 21/5 (a neighbouring node) -> 20/4 is the talo-crural centre.
#   - wrist = 41 (r) / 77 (l): rig 41==42 and 77==78 are coincident (0.00 cm), so the
#     radio-carpien centre is unambiguous.
#   - pelvis = inter-hip span (2 <-> 18).
#   - torso = lumbar 35 -> C7/neck-base 112: geometry-derived, TO VALIDATE downstream.
SEGMENT_SPANS: dict[str, tuple[int, int]] = {
    "femur_r": (18, 19),
    "femur_l": (2, 3),
    "tibia_r": (19, 20),
    "tibia_l": (3, 4),
    "humerus_r": (39, 40),
    "humerus_l": (75, 76),
    "ulna_r": (40, 41),
    "radius_r": (40, 41),
    "ulna_l": (76, 77),
    "radius_l": (76, 77),
    "pelvis": (2, 18),
    "torso": (35, 112),
}

# Bodies with no independent joint span: inherit a parent's scale so the kinematic
# chain and the coupled constraints (patellofemoral, walker_knee, mtp) stay consistent.
# The head has no reliable span and must NOT follow the trunk length -> left at 1.0.
SEGMENT_SCALE_INHERIT: dict[str, str] = {
    "patella_r": "femur_r",
    "patella_l": "femur_l",
    "talus_r": "tibia_r",
    "calcn_r": "tibia_r",
    "toes_r": "tibia_r",
    "talus_l": "tibia_l",
    "calcn_l": "tibia_l",
    "toes_l": "tibia_l",
    "sacrum": "pelvis",
    "hand_r": "radius_r",
    "hand_l": "radius_l",
}


def _span_scale(j0: np.ndarray, js: np.ndarray, prox: int, dist: int) -> float:
    """Subject/template length ratio for a segment (1.0 if template length ~ 0)."""
    length_0 = float(np.linalg.norm(j0[dist] - j0[prox]))
    length_s = float(np.linalg.norm(js[dist] - js[prox]))
    return length_s / length_0 if length_0 > 1e-9 else 1.0


def segment_scales_from_joints(
    j0: np.ndarray, j_subject: np.ndarray, bodies=None
) -> dict[str, tuple[float, float, float]]:
    """Per-body isotropic scale factors from template vs subject rig joints.

    ``j0`` / ``j_subject`` are the (127, 3) template and morphed rig joints. Returns
    ``{opensim_body_name: (sx, sy, sz)}`` with EXACT OpenSim body names as keys.

    If ``bodies`` (an iterable of the model's body names) is given, EVERY one of them is
    present in the result -- bodies with no known span/inheritance get an explicit
    ``(1.0, 1.0, 1.0)`` so the caller can tell "not scaled" from "forgotten". If
    ``bodies`` is ``None``, only the bodies with a known span (+ their inheriting
    children) are returned.
    """
    j0 = np.asarray(j0, dtype=float)
    js = np.asarray(j_subject, dtype=float)
    n = min(j0.shape[0], js.shape[0])

    computed: dict[str, float] = {}
    for body, (prox, dist) in SEGMENT_SPANS.items():
        if 0 <= prox < n and 0 <= dist < n:
            computed[body] = _span_scale(j0, js, prox, dist)
    for child, parent in SEGMENT_SCALE_INHERIT.items():
        if parent in computed:
            computed[child] = computed[parent]

    names = list(bodies) if bodies is not None else list(computed)
    out: dict[str, tuple[float, float, float]] = {}
    for name in names:
        s = computed.get(name, 1.0)
        out[name] = (s, s, s)
    return out


def subject_segment_scales(
    basis: ShapeBasis, betas, *, bodies=None
) -> dict[str, tuple[float, float, float]]:
    """Per-body isotropic skeletal scale factors for a subject's shape ``betas``.

    Morphs the rig joints from ``betas`` (only the scale block moves them) and compares
    to the template (betas = 0). See :func:`segment_scales_from_joints` for the
    ``bodies`` argument and the returned mapping. Feed the result to a ScaleSet.
    """
    sample = morph(basis, betas)
    return segment_scales_from_joints(basis.j0, sample.joint_coords, bodies=bodies)

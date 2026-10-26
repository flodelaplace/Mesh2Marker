"""Stateless generation of a subject-specific .osim with markers on the skin.

Pure core: stdlib + numpy only. No bpy, no MHR/pymomentum. This is the headless
brick of the morphology pipeline: given a source ``.osim``, a (shape-independent)
correspondence map and a subject shape (a morphed or loaded :class:`MhrSample`), it
writes the ``.osim`` with each mapped marker repositioned onto that subject's skin.

Everything is recomputed from the given sample (no stale state): a fresh global
alignment + per-segment transforms, the centroid vertex per marker, and the inverse
segment transform giving the new segment-local marker position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import align_mhr_to_opensim
from .linking import centroid_vertex, reposition_marker_to_vertex
from .mhr import MhrSample
from .morph import ShapeBasis, morph
from .osim import parse_osim, write_osim_with_markers
from .segment_align import ELBOW_ROLL_EXEMPT_MARKERS, compute_segment_transforms

# Wrist markers, KEPT at their model location instead of being reposed onto the skin.
#
# Why freeze rather than map. Skin placement goes through the segment transform, and for the
# forearm that transform has an UNDETERMINED component: the end-to-end fit (elbow -> wrist) pins
# the bone AXIS but not the rotation ABOUT it. At the wrist, whose two markers sit diametrically
# opposite each other, that free rotation is precisely the one that displaces them most. Measured
# on the Synkro model: the markers ended up 75 deg away from the true styloid axis read off the
# model's OWN bone meshes (radius_rv.vtp / ulna_rv.vtp), and the IK rotated the forearm halfway
# trying to satisfy targets no pose could reach.
#
# FOREARM_ROLL_REFERENCE exists to recover that roll but cannot help here, for two separate
# reasons. It is keyed on RWRA/RWRB, absent from this marker set. And even with the right names it
# changes nothing: it aligns the MODEL's marker direction onto the MESH's, yet the model's markers
# are already the output of the faulty path -- a fixed point. The error is only visible against the
# SKELETON, never against a regeneration.
#
# The root cause is anatomical: pronation is NOT a rigid rotation of the forearm. The radius
# crosses over the ulna, the proximal end barely moves and the distal end turns ~90 deg, whereas
# SEGMENT_TABLE treats ulna and radius as one rigid block sharing a single roll. No roll value can
# therefore satisfy the elbow AND the wrist at once.
#
# Freezing sidesteps that and costs little: the styloids are BONY prominences with almost no soft
# tissue, so their position varies little between subjects. Subject size is still honoured -- the
# downstream ScaleTool scales the radius and the markers follow. What is lost is second order;
# what is avoided is a 75 deg error.
WRIST_FROZEN_MARKERS: frozenset[str] = frozenset({
    "RFAradius", "RFAulna", "LFAradius", "LFAulna",   # RGTR / synkro
    "RWRA", "RWRB", "LWRA", "LWRB",                    # Pose2Sim
    "RWR", "RWRM", "LWR", "LWRM",                      # CER / Hagen2026
})
from .segment_scale import segment_scales_from_joints


@dataclass
class SubjectOsim:
    """Result of :func:`generate_subject_osim` when skeleton scales are requested.

    ``marker_locations`` is the same mapping returned by default (marker name ->
    segment-local location). ``segment_scales`` maps every OpenSim body name to an
    isotropic ``(sx, sy, sz)`` skeletal scale factor (1.0 = unscaled), ready for an
    OpenSim ScaleSet -- see :mod:`mesh2marker.segment_scale`.
    """

    marker_locations: dict[str, tuple[float, float, float]]
    segment_scales: dict[str, tuple[float, float, float]]


def frame_scale(matrix) -> float:
    """Isotropic scale of a per-segment similarity (mean column norm of its 3x3 block)."""
    return float(np.linalg.norm(np.asarray(matrix, dtype=float)[:3, :3], axis=0).mean())


def marker_frame_scales(seg_transforms: dict, bodies, anisotropic: dict | None = None
                        ) -> dict[str, tuple[float, float, float]]:
    """Scale of the frame each body's markers were expressed in, per OpenSim body.

    Marker locations are written as ``inv(seg[body] @ world[body]) @ skin_vertex``: they are
    expressed in the body frame AS RESIZED by ``seg[body]``. A downstream fit must give the body
    exactly this scale for the markers to land back on the skin; any other scale moves them off
    it by the ratio (a body scaled 1.36 against a frame of 1.03 pushes its markers 32 % outward).
    Bodies without a transform were not resized (1.0). ``anisotropic`` carries per-axis overrides
    (body frame), e.g. the Hara pelvis.
    """
    aniso = anisotropic or {}
    out: dict[str, tuple[float, float, float]] = {}
    for b in bodies:
        m = seg_transforms.get(b)
        if b in aniso:
            out[b] = aniso[b]
        elif m is None:
            out[b] = (1.0, 1.0, 1.0)
        else:
            s = frame_scale(m)
            out[b] = (s, s, s)
    return out


def pelvis_hara2016(seg_transforms: dict, model, sample: MhrSample, global_transform
                    ) -> dict[str, tuple[float, float, float]]:
    """Pelvis WIDTH from Hara et al. 2016 instead of the global bone scale (edits ``seg_transforms``).

    Hara R, McGinley J, Briggs C, Baker R, Sangeux M. Predicting the location of the hip joint
    centres, impact of age group and sex. Sci Rep 2016;6:37707 (post-mortem CT, n = 157): the
    medio-lateral hip-centre coordinate from mid-ASIS is 8 + 0.086 LL mm, LL = bony leg length
    ASIS -> medial epicondyle -> medial malleolus, so the inter-hip-centre distance is
    2 (8 + 0.086 LL). LL comes from the SAME long-bone fit as the rest of the skeleton (femur +
    tibia rig spans) plus the ASIS -> hip-centre height 9 + 0.078 LL of the same paper:
    LL = (9 + F + T) / (1 - 0.078). Never from the skin, so it does not grow with adiposity
    (Brynskog et al. 2021, J Biomech, CT, BMI 28: pelvic bone size barely depends on BMI).

    The pelvis keeps the rotation and origin placement of its landmark fit; its frame gets the
    anisotropic scale (s_bone, s_bone, s_width) in the OpenSim pelvis frame (z = medio-lateral),
    s_bone = mean femur/tibia frame scale. The sacrum, rigid with it, follows. Returns the
    per-axis scales of the overridden bodies ({} if the inputs are missing).
    """
    from .kinematics import forward_kinematics, joint_centers
    from .segment_align import SEGMENT_TABLE

    if "pelvis" not in seg_transforms:
        return {}
    centers = joint_centers(model)

    def span(body: str) -> float | None:
        row = SEGMENT_TABLE.get(body)
        if row is None:
            return None
        _, _, mp, md = row
        a = np.asarray(global_transform.apply(np.asarray(sample.joint_coords[mp], float)), float)
        b = np.asarray(global_transform.apply(np.asarray(sample.joint_coords[md], float)), float)
        return float(np.linalg.norm(b - a))

    F = [span("femur_r"), span("femur_l")]
    T = [span("tibia_r"), span("tibia_l")]
    if None in F + T or "hip_r" not in centers or "hip_l" not in centers:
        return {}
    LL = (9.0 + 1000.0 * (np.mean(F) + np.mean(T))) / (1.0 - 0.078)  # mm
    width = 2.0 * (8.0 + 0.086 * LL) / 1000.0  # m
    w0 = float(np.linalg.norm(np.asarray(centers["hip_r"]) - np.asarray(centers["hip_l"])))
    longs = [frame_scale(seg_transforms[b]) for b in ("femur_r", "femur_l", "tibia_r", "tibia_l")
             if b in seg_transforms]
    s_bone = float(np.mean(longs)) if longs else frame_scale(seg_transforms["pelvis"])
    D = np.diag([s_bone, s_bone, width / w0])
    S = np.asarray(seg_transforms["pelvis"], float)
    Rn = S[:3, :3] / frame_scale(S)
    wb = forward_kinematics(model)["pelvis"]
    Rb, ob = np.asarray(wb.rotation, float), np.asarray(wb.translation, float)
    body = np.eye(4)
    body[:3, :3] = Rn @ Rb @ D
    body[:3, 3] = S[:3, :3] @ ob + S[:3, 3]  # the pelvis origin lands where the landmark fit put it
    W = np.eye(4)
    W[:3, :3], W[:3, 3] = Rb, ob
    seg = body @ np.linalg.inv(W)
    out = {}
    for b in ("pelvis", "sacrum"):
        if b in seg_transforms:
            seg_transforms[b] = seg.copy()
            out[b] = (s_bone, s_bone, float(width / w0))
    return out


def _as_index_map(correspondence) -> dict[str, list[int]]:
    """Accept either a CorrespondenceFile or a plain {marker: indices} mapping."""
    if hasattr(correspondence, "markers"):
        return {m.name: list(m.mhr_vertices) for m in correspondence.markers}
    return {name: list(indices) for name, indices in correspondence.items()}


def generate_subject_osim(
    src_osim_path,
    dst_osim_path,
    correspondence,
    sample: MhrSample,
    *,
    shape_basis: ShapeBasis | None = None,
    alignment_markers=None,
    frozen_markers=(),
    pelvis: str | None = None,
    frame_scales: bool = False,
):
    """Write ``dst`` = ``src`` with mapped markers moved onto ``sample``'s skin.

    Pure function: parses the source model, aligns the sample to it, computes the
    per-segment transforms, and for every mapped marker present in the model with at
    least one vertex index, takes the centroid vertex and inverts the segment world
    matrix to get the new segment-local location.

    The correspondence also drives the per-segment alignment (``landmark_map``), so a
    model whose marker names differ from the Pose2Sim vocabulary (e.g. SKEL) gets its
    head/hands/pelvis/torso aligned from its own map -- matching the Blender add-on.

    ``frozen_markers`` (optional) lists markers KEPT at their model location instead of
    being repositioned onto the skin -- see :data:`WRIST_FROZEN_MARKERS` for why the wrist
    pair belongs there. They still drive the per-segment alignment; only their own
    replacement is skipped. Empty by default, so existing callers are unchanged.

    ``alignment_markers`` (optional) restricts WHICH mapped markers drive that
    per-segment alignment; every mapped marker is still repositioned onto the skin.
    Use it when the map carries markers that are not anatomical IK targets -- e.g. the
    plantar ``SOLE_*`` contact markers, soft-tissue points added for ground-contact
    handling: letting them into the Procrustes pulls the segment off its bony landmarks
    (measured: the calcaneus fit degrades from 1.4 cm to 2.1 cm at the heel marker).
    ``None`` (default) keeps the legacy behaviour -- every mapped marker aligns.

    Elbow markers (``ELBOW_ROLL_EXEMPT_MARKERS``) are placed from the forearm's PRE-roll
    frame: they mark the humeral epicondyle, which does not pronate, so they must not
    ride the forearm roll that orients the wrist/hand.

    Return value (opt-in, backward compatible):

    - ``shape_basis`` omitted (default) -> returns the ``{marker: location}`` dict, as
      before. The written ``.osim`` keeps the source skeleton (bones NOT scaled).
    - ``frame_scales=True`` -> returns a :class:`SubjectOsim` whose ``segment_scales`` are the
      scales of the frames the markers were EXPRESSED in (see :func:`marker_frame_scales`): the
      ones a downstream fit must apply for the markers to sit on the skin. Takes precedence over
      ``shape_basis``.
    - ``pelvis="hara2016"`` -> pelvis width from Hara et al. 2016 on the fitted leg length
      instead of the global bone scale (see :func:`pelvis_hara2016`); pelvis markers are
      expressed for that pelvis.
    - ``shape_basis`` given -> returns a :class:`SubjectOsim` carrying both the marker
      locations AND ``segment_scales`` (every model body -> isotropic ``(sx, sy, sz)``,
      1.0 = unscaled) read from the subject's rig joints. The scales are for an OpenSim
      ScaleSet downstream; this function still does not scale the ``.osim`` itself.
    """
    index_map = _as_index_map(correspondence)
    model = parse_osim(src_osim_path)
    global_transform, _, _ = align_mhr_to_opensim(sample, model)
    landmark_map = {
        name: centroid_vertex(sample.verts, list(indices))
        for name, indices in index_map.items()
        if indices
    }
    if alignment_markers is not None:
        allowed = set(alignment_markers)
        landmark_map = {n: v for n, v in landmark_map.items() if n in allowed}
    seg_transforms = compute_segment_transforms(
        sample, model, global_transform, landmark_map=landmark_map
    )
    aniso: dict = {}
    if pelvis == "hara2016":
        aniso = pelvis_hara2016(seg_transforms, model, sample, global_transform)
    elif pelvis is not None:
        raise ValueError(f"unknown pelvis option {pelvis!r} (expected None or 'hara2016')")
    # Lazily built only if an elbow-exempt marker is actually present: the same
    # transforms but with the forearm roll disabled, so epicondyle markers stay put.
    seg_no_roll: dict | None = None

    model_markers = {m.name for m in model.markers}
    locations: dict[str, tuple[float, float, float]] = {}
    frozen = set(frozen_markers or ())
    model_location = {m.name: tuple(float(c) for c in m.location) for m in model.markers}
    for marker_name, indices in index_map.items():
        if not indices or marker_name not in model_markers:
            continue
        if marker_name in frozen:
            # Keeps its model location: see WRIST_FROZEN_MARKERS. It stays in landmark_map, so it
            # still orients the segment; only its own skin REPLACEMENT is skipped. It is still
            # RETURNED, at the model value: a marker missing from the dict would be
            # indistinguishable from one that was silently dropped.
            locations[marker_name] = model_location[marker_name]
            continue
        idx = centroid_vertex(sample.verts, list(indices))
        seg = seg_transforms
        if marker_name in ELBOW_ROLL_EXEMPT_MARKERS:
            if seg_no_roll is None:
                seg_no_roll = compute_segment_transforms(
                    sample, model, global_transform,
                    landmark_map=landmark_map, apply_forearm_roll=False,
                )
            seg = seg_no_roll
        local = reposition_marker_to_vertex(
            model, marker_name, sample.verts, idx, global_transform, seg
        )
        locations[marker_name] = (float(local[0]), float(local[1]), float(local[2]))

    write_osim_with_markers(src_osim_path, dst_osim_path, locations)
    if frame_scales:
        return SubjectOsim(
            marker_locations=locations,
            segment_scales=marker_frame_scales(seg_transforms, [b.name for b in model.bodies], aniso),
        )
    if shape_basis is None:
        return locations
    segment_scales = segment_scales_from_joints(
        shape_basis.j0, sample.joint_coords, bodies=[b.name for b in model.bodies]
    )
    return SubjectOsim(marker_locations=locations, segment_scales=segment_scales)


def sample_from_betas(basis: ShapeBasis, betas) -> MhrSample:
    """Sugar: the morphed rest-pose sample for the given shape coefficients."""
    return morph(basis, betas)

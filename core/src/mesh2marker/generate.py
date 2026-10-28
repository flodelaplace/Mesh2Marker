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


def impose_body_scale(seg_transforms: dict, model, body: str, scale) -> None:
    """Give ``body`` the (possibly anisotropic) scale ``scale`` in ITS OWN frame (edits ``seg_transforms``).

    Keeps what the fit decided about the segment's PLACEMENT (its rotation and where its origin lands on
    the subject) and replaces only its size. The per-body correction is a world-space map applied on top
    of the neutral pose (``seg @ world[body]``); the new one sends a body-local point p to
    ``Rn Rb diag(scale) p + seg(origin)``, Rn the fit's rotation, Rb/origin the neutral body frame.
    """
    from .kinematics import forward_kinematics

    S = np.asarray(seg_transforms.get(body, np.eye(4)), float)
    Rn = S[:3, :3] / frame_scale(S)
    wb = forward_kinematics(model)[body]
    Rb, ob = np.asarray(wb.rotation, float), np.asarray(wb.translation, float)
    placed = np.eye(4)
    placed[:3, :3] = Rn @ Rb @ np.diag(np.asarray(scale, float).reshape(3))
    placed[:3, 3] = S[:3, :3] @ ob + S[:3, 3]
    W = np.eye(4)
    W[:3, :3], W[:3, 3] = Rb, ob
    seg_transforms[body] = placed @ np.linalg.inv(W)


def pelvis_hara2016(seg_transforms: dict, model, sample: MhrSample, global_transform,
                    long_scales: dict | None = None) -> dict[str, tuple[float, float, float]]:
    """Pelvis WIDTH from Hara et al. 2016 instead of the global bone scale (edits ``seg_transforms``).

    Hara R, McGinley J, Briggs C, Baker R, Sangeux M. Predicting the location of the hip joint
    centres, impact of age group and sex. Sci Rep 2016;6:37707 (post-mortem CT, n = 157): the
    medio-lateral hip-centre coordinate from mid-ASIS is 8 + 0.086 LL mm, LL = bony leg length
    ASIS -> medial epicondyle -> medial malleolus, so the inter-hip-centre distance is
    2 (8 + 0.086 LL), with the ASIS -> hip-centre height 9 + 0.078 LL of the same paper:
    LL = (9 + F + T) / (1 - 0.078). F, T (femur, tibia) come from the SAME skeleton as the rest:
    the subject's rig spans, or ``long_scales`` (body -> scale, e.g. imposed from outside) times the
    model's own bone lengths. Never from the skin, so it does not grow with adiposity (Brynskog et al.
    2021, J Biomech, CT, BMI 28: pelvic bone size barely depends on BMI).

    The pelvis keeps the rotation and origin placement of its fit and gets the anisotropic scale
    (s_bone, s_bone, s_width) in the OpenSim pelvis frame (z = medio-lateral), s_bone = mean femur/tibia
    scale. The sacrum, rigid with it, takes the same world map. Returns the per-axis scales set.
    """
    from .kinematics import joint_centers
    from .segment_align import SEGMENT_TABLE

    if "pelvis" not in seg_transforms:
        return {}
    centers = joint_centers(model)

    def span(body: str) -> float | None:
        row = SEGMENT_TABLE.get(body)
        if row is None:
            return None
        jp, jd, mp, md = row
        if long_scales is not None and body in long_scales and jp in centers and jd in centers:
            L0 = float(np.linalg.norm(np.asarray(centers[jd]) - np.asarray(centers[jp])))
            return L0 * float(np.mean(long_scales[body]))
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
    if long_scales is not None:
        longs = [float(np.mean(long_scales[b])) for b in ("femur_r", "femur_l", "tibia_r", "tibia_l")
                 if b in long_scales]
    else:
        longs = [frame_scale(seg_transforms[b]) for b in ("femur_r", "femur_l", "tibia_r", "tibia_l")
                 if b in seg_transforms]
    s_bone = float(np.mean(longs)) if longs else frame_scale(seg_transforms["pelvis"])
    sc = (s_bone, s_bone, float(width / w0))
    impose_body_scale(seg_transforms, model, "pelvis", sc)
    out = {"pelvis": sc}
    if "sacrum" in seg_transforms:
        seg_transforms["sacrum"] = seg_transforms["pelvis"].copy()
        out["sacrum"] = sc
    return out


def pelvis_hara2016_3d(seg_transforms: dict, model, sample: MhrSample, global_transform,
                       landmark_map: dict) -> tuple[dict[str, tuple[float, float, float]], np.ndarray | None]:
    """Hip-joint centres at the HEIGHT given by Hara et al. 2016, femur sized from them (edits ``seg_transforms``).

    The rig's hip centres are not anatomical, and the femur measured from them inherits the error. Hara et al. 2016
    (Sci Rep 6:37707, CT, n = 157) give the hip centre in the conventional-gait-model pelvis frame (origin mid-ASIS,
    X anterior, Y lateral, Z superior), in mm, from the leg length LL alone: X = 11 - 0.063 LL, Y = 8 + 0.086 LL,
    Z = -9 - 0.078 LL. LL is measured as in the clinic, on the SKIN of the subject's rest body: ASIS -> medial femoral
    condyle -> medial malleolus (mean of both sides).

    The pelvis keeps its shape (width from Hara, :func:`pelvis_hara2016`) and the rotation of its fit; it is moved
    along the subject's pelvic vertical so that its mid hip centre sits Z below the subject's mid-ASIS. Only the
    height: the ASIS skin of a corpulent subject is far in front of the bone (antero-posterior), barely above it.
    Each femur then gets the isotropic scale of its new span hip centre -> rig knee centre. Returns the scales set
    and the world mid hip centre (the trunk is measured from the SAME start point).
    """
    from .kinematics import joint_centers
    from .segment_align import SEGMENT_TABLE, minimal_rotation

    need = ("RASI", "LASI", "RPSI", "LPSI", "RMFC", "LMFC", "RMMAL", "LMMAL")
    c = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    if "pelvis" not in seg_transforms or not all(n in landmark_map for n in need) or "hip_r" not in c \
            or "hip_l" not in c:
        return {}, None
    V = lambda n: np.asarray(global_transform.apply(np.asarray(sample.verts[landmark_map[n]], float)), float)  # noqa: E731
    LL = 1000.0 * float(np.mean([np.linalg.norm(V(a) - V(k)) + np.linalg.norm(V(k) - V(m))
                                  for a, k, m in (("RASI", "RMFC", "RMMAL"), ("LASI", "LMFC", "LMMAL"))]))
    Zh = (-9.0 - 0.078 * LL) / 1000.0
    out = pelvis_hara2016(seg_transforms, model, sample, global_transform)
    if not out:
        return {}, None
    S = seg_transforms["pelvis"]
    # vertical of the subject's pelvis (CGM frame from the skin ASIS / PSIS)
    asis = (V("RASI") + V("LASI")) / 2
    psis = (V("RPSI") + V("LPSI")) / 2
    lat = V("RASI") - V("LASI")
    lat /= np.linalg.norm(lat)
    ant = asis - psis
    ant -= (ant @ lat) * lat
    ant /= np.linalg.norm(ant)
    up = np.cross(lat, ant)  # z = x cross y in the CGM frame (x ant, y lat right -> z sup ... sign checked below)
    if up[1] < 0:
        up = -up
    hip = lambda: (S[:3, :3] @ ((c["hip_r"] + c["hip_l"]) / 2) + S[:3, 3])  # noqa: E731
    dz = float((asis + Zh * up - hip()) @ up)
    S[:3, 3] = S[:3, 3] + dz * up
    if "sacrum" in seg_transforms:
        seg_transforms["sacrum"] = S.copy()
    hips = {sd: S[:3, :3] @ c[f"hip_{sd}"] + S[:3, 3] for sd in ("r", "l")}
    for sd in ("r", "l"):
        body = f"femur_{sd}"
        row = SEGMENT_TABLE.get(body)
        if body not in seg_transforms or row is None or row[1] not in c:
            continue
        knee = np.asarray(global_transform.apply(np.asarray(sample.joint_coords[row[3]], float)), float)
        L0 = float(np.linalg.norm(c[row[1]] - c[f"hip_{sd}"]))
        s_f = float(np.linalg.norm(knee - hips[sd]) / L0)
        impose_body_scale(seg_transforms, model, body, (s_f, s_f, s_f))
        out[body] = (s_f,) * 3
        # re-viser le femur : son axe va de la hanche Hara au genou du rig (il partait de la hanche du rig)
        F = np.array(seg_transforms[body], float)
        cur = F[:3, :3] @ (c[row[1]] - c[f"hip_{sd}"])
        Rv = minimal_rotation(cur / np.linalg.norm(cur), (knee - hips[sd]) / np.linalg.norm(knee - hips[sd]))
        F[:3, :3] = Rv @ F[:3, :3]
        F[:3, 3] = hips[sd] - F[:3, :3] @ c[f"hip_{sd}"]
        seg_transforms[body] = F
    rig_h = np.mean([np.asarray(global_transform.apply(np.asarray(sample.joint_coords[SEGMENT_TABLE[f"femur_{sd}"][2]],
                                                                  float)), float) for sd in ("r", "l")], axis=0)
    print(f"  [hara 3D] LL {LL:.0f} mm ; hanches {1000 * Zh:.0f} mm sous les ASIS (rig : "
          f"{1000 * float((rig_h - asis) @ up):.0f}) ; bassin deplace de {1000 * dz:+.0f} mm ; "
          f"femur {round(out.get('femur_r', (0,))[0], 3)}")
    out["_hips"] = hips
    return out, (hips["r"] + hips["l"]) / 2


def foot_from_sole(seg_transforms: dict, model, sample: MhrSample, global_transform,
                   landmark_map: dict) -> dict[str, tuple[float, float, float]]:
    """Foot sized on the SOLE so it sits flat under the skin (edits ``seg_transforms``).

    Measured on the subject's rest body and on the model, side by side: the ANKLE HEIGHT above the plantar plane (the
    sole markers SOLE_*) and the FOOT LENGTH (heel-most to toe-most sole marker, along the foot). Talus, calcaneus and
    toes get (length ratio, height ratio, length ratio) in the OpenSim foot frame (x forward, y up, z lateral). With the
    ankle on the rig's ankle centre, the bony sole then lands on the skin sole instead of floating ~2 cm above it (P16,
    least-squares placement, foot scaled x1.0 or by stature).
    """
    from .kinematics import joint_centers
    from .markers import neutral_marker_world_positions
    from .segment_align import SEGMENT_TABLE

    c = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    mw = neutral_marker_world_positions(model)
    V = lambda n: np.asarray(global_transform.apply(np.asarray(sample.verts[landmark_map[n]], float)), float)  # noqa: E731
    out: dict = {}
    for sd in ("r", "l"):
        sole = [n for n in landmark_map if n.startswith("SOLE") and n.endswith(f"_{sd}") and n in mw]
        ank = f"ankle_{sd}"
        if len(sole) < 3 or ank not in c:
            continue
        Ps = np.array([V(n) for n in sole]); Pm = np.array([mw[n] for n in sole])
        J = np.asarray(global_transform.apply(np.asarray(sample.joint_coords[SEGMENT_TABLE[f"tibia_{sd}"][3]], float)), float)
        h_s = float(J[1] - Ps[:, 1].min()); h_m = float(c[ank][1] - Pm[:, 1].min())
        L_s = float(np.ptp(Ps[:, 0])); L_m = float(np.ptp(Pm[:, 0]))
        if min(h_s, h_m, L_s, L_m) <= 0:
            continue
        sc = (L_s / L_m, h_s / h_m, L_s / L_m)
        for bd in (f"talus_{sd}", f"calcn_{sd}", f"toes_{sd}"):
            if bd in seg_transforms:
                impose_body_scale(seg_transforms, model, bd, sc)
                out[bd] = tuple(float(x) for x in sc)
        print(f"  [pied par la semelle] {sd} : cheville {100 * h_s:.1f} cm au-dessus de la semelle (modele {100 * h_m:.1f}), "
              f"longueur {100 * L_s:.1f} cm (modele {100 * L_m:.1f}) -> echelle {[round(x, 3) for x in sc]}")
    return out


TRUNK_BODIES = ("lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", "Abdomen", "torso")


def trunk_after_pelvis(seg_transforms: dict, model, sample: MhrSample, global_transform,
                       pelvis_scale, target_span: float | None = None,
                       hip_mid: np.ndarray | None = None) -> dict[str, tuple[float, float, float]]:
    """Trunk scale computed AFTER the pelvis, so the skeleton keeps its hip-to-shoulder height (edits ``seg_transforms``).

    The hip-to-shoulder span crosses the pelvis (hips -> L5/S1) then the lumbar column and torso. Its length is
    the subject's (mid-hip -> mid-shoulder of the rig at rest). With the pelvis already sized (``pelvis_scale``,
    body frame), the trunk bodies get ONE isotropic scale s solving
    ``| diag(pelvis_scale) (B - H) + s (S - B) | = D_subject``, H/B/S = mid-hip, L5/S1 and mid-shoulder of the
    neutral model. Sizing the pelvis without this left the shoulders ~3 cm low (P16), and the downstream fit
    inflated the head (x1.16 -> x1.30) to reach the head markers.
    """
    from .kinematics import joint_centers
    from .segment_align import SEGMENT_TABLE

    c = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    need = ("hip_r", "hip_l", "L5_S1_IVDjnt", "acromial_r", "acromial_l")
    if not all(k in c for k in need):
        return {}
    H = (c["hip_r"] + c["hip_l"]) / 2
    B = c["L5_S1_IVDjnt"]
    S = (c["acromial_r"] + c["acromial_l"]) / 2
    J = lambda i: np.asarray(global_transform.apply(np.asarray(sample.joint_coords[i], float)), float)  # noqa: E731
    hips = [SEGMENT_TABLE["femur_r"][2], SEGMENT_TABLE["femur_l"][2]]
    shoulders = [SEGMENT_TABLE["humerus_r"][2], SEGMENT_TABLE["humerus_l"][2]]
    H_s = (J(hips[0]) + J(hips[1])) / 2 if hip_mid is None else np.asarray(hip_mid, float)  # depart : hanches Hara
    D = float(np.linalg.norm((J(shoulders[0]) + J(shoulders[1])) / 2 - H_s))
    if target_span is not None:  # hauteur hanches -> epaules donnee (ex. centres articulaires de l'essai)
        D = float(target_span)
    u = np.diag(np.asarray(pelvis_scale, float).reshape(3)) @ (B - H)
    v = S - B
    a, b, cc = float(v @ v), float(2 * u @ v), float(u @ u - D * D)
    disc = b * b - 4 * a * cc
    if a <= 0 or disc < 0:
        return {}
    s_t = (-b + np.sqrt(disc)) / (2 * a)
    out = {}
    for bd in TRUNK_BODIES:
        if bd in {x.name for x in model.bodies}:
            impose_body_scale(seg_transforms, model, bd, (s_t, s_t, s_t))
            out[bd] = (float(s_t),) * 3
    return out


def trunk_from_acromions(seg_transforms: dict, model, sample: MhrSample, global_transform, pelvis_scale,
                         landmark_map: dict, target_span: float | None = None,
                         acromions: tuple[str, str] = ("RACR", "LACR")) -> dict[str, tuple[float, float, float]]:
    """Trunk sized on the ACROMIONS, after the pelvis (edits ``seg_transforms``).

    The rig's shoulder joints are not anatomical (P16: 35.9 cm apart where the opto-scaled reference model puts the
    glenohumeral centres 40.0 cm apart). The acromions are subcutaneous bony landmarks -- little soft tissue whatever
    the corpulence -- so they size the trunk instead:
      - WIDTH (torso z, medio-lateral): subject inter-acromion distance / the model's own acromion markers ;
      - HEIGHT (torso x, y and the lumbar column, isotropic): solved, after the pelvis, so that mid-hip -> mid-acromion
        of the model equals the subject's: ``| diag(pelvis_scale)(B - H) + diag(s, s, w)(A - B) | = D``, H/B/A =
        mid-hip, L5/S1, mid-acromion of the neutral model (A - B is ~ sagittal, so w barely enters).
    ``target_span`` overrides D (e.g. the walking median, the rest pose having the shoulders ~1 cm higher). The shoulder
    joints then sit where the model's geometry puts them relative to the acromions.
    """
    from .kinematics import joint_centers
    from .markers import neutral_marker_world_positions
    from .segment_align import SEGMENT_TABLE

    c = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    mw = neutral_marker_world_positions(model)
    ar, al = acromions
    if not all(k in c for k in ("hip_r", "hip_l", "L5_S1_IVDjnt")) or ar not in mw or al not in mw \
            or ar not in landmark_map or al not in landmark_map:
        return {}
    V = lambda name: np.asarray(global_transform.apply(np.asarray(sample.verts[landmark_map[name]], float)), float)  # noqa: E731
    J = lambda i: np.asarray(global_transform.apply(np.asarray(sample.joint_coords[i], float)), float)  # noqa: E731
    w = float(np.linalg.norm(V(ar) - V(al)) / np.linalg.norm(mw[ar] - mw[al]))
    hips = [SEGMENT_TABLE["femur_r"][2], SEGMENT_TABLE["femur_l"][2]]
    D = float(np.linalg.norm((V(ar) + V(al)) / 2 - (J(hips[0]) + J(hips[1])) / 2))
    if target_span is not None:
        D = float(target_span)
    H = (c["hip_r"] + c["hip_l"]) / 2
    B = c["L5_S1_IVDjnt"]
    A = (mw[ar] + mw[al]) / 2
    u = np.diag(np.asarray(pelvis_scale, float).reshape(3)) @ (B - H)
    v = A - B
    vh, vw = np.array([v[0], v[1], 0.0]), np.array([0.0, 0.0, v[2]])  # height part scales with s, width part with w
    u2 = u + w * vw
    a, b, cc = float(vh @ vh), float(2 * u2 @ vh), float(u2 @ u2 - D * D)
    disc = b * b - 4 * a * cc
    if a <= 0 or disc < 0:
        return {}
    s_h = float((-b + np.sqrt(disc)) / (2 * a))
    out = {}
    names = {x.name for x in model.bodies}
    for bd in TRUNK_BODIES:
        if bd not in names:
            continue
        sc = (s_h, s_h, w) if bd in ("torso", "Abdomen") else (s_h, s_h, s_h)
        impose_body_scale(seg_transforms, model, bd, sc)
        out[bd] = tuple(float(x) for x in sc)
    return out


def trunk_from_skin_height(seg_transforms: dict, model, sample: MhrSample, global_transform, pelvis_scale,
                           landmark_map: dict,
                           pelvis_markers: tuple[str, ...] = ("RASI", "LASI", "RPSI", "LPSI"),
                           acromions: tuple[str, str] = ("RACR", "LACR")) -> dict[str, tuple[float, float, float]]:
    """Trunk HEIGHT from skin landmarks, after the pelvis (edits ``seg_transforms``).

    Starting point and end point are the same on the subject and on the model: the mid of the four iliac spines and
    the mid of the acromions -- not the rig's hip centres, which are not anatomical (the pelvis is sized by Hara, so
    a span measured from the rig hips would sit the trunk on a different base). Only the VERTICAL component is used:
    soft tissue thickens the body mostly forwards and sideways, so the iliac spines of a corpulent subject move little
    in height. With the pelvis already sized (``pelvis_scale``), one isotropic scale s for the lumbar column and torso
    solves ``Dy_subject = ps_y (B - P)_y + s (A - B)_y`` (P, B, A = mid iliac spines, L5/S1, mid acromions of the
    neutral model). The width is left to that scale (no separate medio-lateral factor).
    """
    from .kinematics import joint_centers
    from .markers import neutral_marker_world_positions

    c = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    mw = neutral_marker_world_positions(model)
    need = tuple(pelvis_markers) + tuple(acromions)
    if "L5_S1_IVDjnt" not in c or not all(n in mw and n in landmark_map for n in need):
        return {}
    V = lambda name: np.asarray(global_transform.apply(np.asarray(sample.verts[landmark_map[name]], float)), float)  # noqa: E731
    mid = lambda f, names: np.mean([f(n) for n in names], axis=0)  # noqa: E731
    Dy = float(mid(V, acromions)[1] - mid(V, pelvis_markers)[1])
    P, B, A = mid(mw.__getitem__, pelvis_markers), c["L5_S1_IVDjnt"], mid(mw.__getitem__, acromions)
    ps_y = float(np.asarray(pelvis_scale, float).reshape(3)[1])
    den = float((A - B)[1])
    if den <= 0:
        return {}
    s_h = (Dy - ps_y * float((B - P)[1])) / den
    if not np.isfinite(s_h) or s_h <= 0:
        return {}
    out = {}
    names = {x.name for x in model.bodies}
    for bd in TRUNK_BODIES:
        if bd in names:
            impose_body_scale(seg_transforms, model, bd, (s_h, s_h, s_h))
            out[bd] = (float(s_h),) * 3
    return out


def chain_consistent(seg_transforms: dict, model, root: str = "pelvis",
                     stop: tuple[str, ...] = ("humerus_r", "humerus_l")) -> None:
    """Re-place every segment along the KINEMATIC CHAIN (edits ``seg_transforms``).

    The per-segment fit places each body independently on the subject (long bones end to end, pelvis/torso/head by
    landmark Procrustes). Marker locations are then expressed against those free placements, but in OpenSim the
    bodies form a chain: a child sits where its parent's joint puts it. When the two disagree, markers land off the
    skin once the model is assembled (P16: torso ~2 cm low, the downstream fit inflated the head to compensate).
    Here each body keeps the ROTATION and SIZE of its fit, and its translation is recomputed top-down from ``root``
    so that its joint centre lands exactly where its (already placed) parent sends that joint.

    The chain stops at ``stop`` (default: the arms). The arm bones keep their fit on the rig's elbow and wrist
    centres: hanging the humerus from the model's shoulder moved it ~6 cm off the subject's arm (P16), and the IK then
    rotated the forearm and hand around a wrong axis.
    """
    from .kinematics import joint_centers

    centers = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    children: dict[str, list] = {}
    for j in model.joints:
        children.setdefault(j.parent_body, []).append(j)
    queue = [root]
    seen = {root}
    while queue:
        parent = queue.pop(0)
        sp = np.asarray(seg_transforms.get(parent, np.eye(4)), float)
        for j in children.get(parent, []):
            c = j.child_body
            if c in seen or j.name not in centers or c in stop:
                continue
            seen.add(c)
            target = sp[:3, :3] @ centers[j.name] + sp[:3, 3]
            sc = np.array(seg_transforms.get(c, np.eye(4)), dtype=float)
            sc[:3, 3] = target - sc[:3, :3] @ centers[j.name]
            seg_transforms[c] = sc
            queue.append(c)


def place_least_squares(seg_transforms: dict, model, sample: MhrSample, global_transform,
                        hip_targets: dict | None = None, w_joint: float = 1000.0, w_target: float = 10.0,
                        w_prior: float = 0.01, free_joints: tuple[str, ...] = ("acromial_r", "acromial_l")) -> dict:
    """Place every segment at once by weighted least squares (edits ``seg_transforms``).

    Each body keeps the ROTATION and SIZE of its fit (so the marker logic, forearm roll included, is untouched);
    only the translations are solved, jointly, so that:
      - parent and child agree on each joint centre (weight ``w_joint``, near hard);
      - the reliable centres land on their targets (``w_target``): the rig's knees, ankles, elbows, wrists, and
        the hip centres given in ``hip_targets`` (Hara) when present;
      - each body stays near its own fit (``w_prior``, fixes what nothing else constrains).
    Unlike :func:`chain_consistent` (top-down), an error anywhere is shared out instead of being carried down to the
    feet. The shoulders (``free_joints``) are not tied: the arms keep their fit on the rig's elbow and wrist.
    Returns the residual (m) per target joint.
    """
    from .kinematics import joint_centers
    from .segment_align import SEGMENT_TABLE

    centers = {k: np.asarray(v, float) for k, v in joint_centers(model).items()}
    bodies = [b.name for b in model.bodies if b.name in seg_transforms]
    idx = {b: i for i, b in enumerate(bodies)}
    n = len(bodies)
    J = lambda i: np.asarray(global_transform.apply(np.asarray(sample.joint_coords[i], float)), float)  # noqa: E731
    targets: dict[str, np.ndarray] = {}
    for jp, jd, mp, md in SEGMENT_TABLE.values():
        for jn, mi in ((jp, mp), (jd, md)):
            if jn.startswith(("walker_knee", "ankle", "elbow", "radius_hand")):
                targets[jn] = J(mi)
    for jn, t in (hip_targets or {}).items():
        targets[jn] = np.asarray(t, float)
    A = {b: np.asarray(seg_transforms[b], float)[:3, :3] for b in bodies}
    rows, rhs = [], []

    def eq(coefs: dict, const: np.ndarray, w: float) -> None:
        for k in range(3):
            r = np.zeros(3 * n)
            for b, c in coefs.items():
                r[3 * idx[b] + k] = c * w
            rows.append(r)
            rhs.append(-const[k] * w)

    target_body: dict[str, str] = {}
    for j in model.joints:
        p, c = j.parent_body, j.child_body
        if j.name not in centers or c not in idx:
            continue
        X = centers[j.name]
        if j.name in targets:
            target_body[j.name] = c
            eq({c: 1.0}, A[c] @ X - targets[j.name], w_target)
        if p in idx and j.name not in free_joints:
            eq({p: 1.0, c: -1.0}, A[p] @ X - A[c] @ X, w_joint)
    for b in bodies:
        eq({b: 1.0}, -np.asarray(seg_transforms[b], float)[:3, 3], w_prior)
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    for b in bodies:
        S = np.array(seg_transforms[b], float)
        S[:3, 3] = sol[3 * idx[b]:3 * idx[b] + 3]
        seg_transforms[b] = S
    return {jn: float(np.linalg.norm(A[b] @ centers[jn] + sol[3 * idx[b]:3 * idx[b] + 3] - targets[jn]))
            for jn, b in target_body.items()}


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
    imposed_scales: dict | None = None,
    trunk: str | None = None,
    trunk_span: float | None = None,
    global_scale: bool = True,
    chain: bool | str = False,
    global_first: bool = False,
    foot: str | None = None,
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
    - ``imposed_scales`` ({body: (sx, sy, sz)}) -> those bodies keep the fit's placement but take
      the given size, and their markers are expressed for it (one scale source for the whole chain).
    - ``chain=True`` -> segments re-placed along the kinematic chain (rotation and size of their fit, translation
      from their parent's joint), see :func:`chain_consistent`.
    - ``global_scale=False`` -> rigid global alignment (subject given at true metric size): segment
      scales are absolute sizes rather than proportions.
    - ``trunk="after_pelvis"`` -> lumbar column + torso scaled AFTER the pelvis so the hip-to-shoulder
      height of the subject is kept (see :func:`trunk_after_pelvis`).
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
    # global_scale=False : calage RIGIDE (rotation + translation) du sujet sur le modele. A utiliser quand le sujet
    # est donne a sa vraie taille (metres) : les echelles par segment deviennent alors des tailles ABSOLUES, au lieu de
    # proportions apres une similitude globale qui ramene tout le corps a la taille du modele.
    global_transform, _, _ = align_mhr_to_opensim(sample, model, with_scale=global_scale)
    landmark_map = {
        name: centroid_vertex(sample.verts, list(indices))
        for name, indices in index_map.items()
        if indices
    }
    _full_landmarks = dict(landmark_map)  # tous les reperes de la carte (le tronc par les acromions s'en sert)
    if alignment_markers is not None:
        allowed = set(alignment_markers)
        landmark_map = {n: v for n, v in landmark_map.items() if n in allowed}
    seg_transforms = compute_segment_transforms(
        sample, model, global_transform, landmark_map=landmark_map
    )
    _ctx: dict = {}

    def _ajuster(seg: dict) -> dict:
        """Imposed sizes, Hara pelvis, trunk after pelvis, kinematic chain -- the SAME for every transform set."""
        aniso: dict = {}
        _ctx.clear()
        if global_first:
            # « ScaleTool global » d'abord : le modele entier a la stature du sujet ; seuls les segments sans mesure
            # propre (pieds, mains) gardent ce facteur, les autres sont ensuite mis a leur taille mesuree
            from .markers import neutral_marker_world_positions
            mw0 = neutral_marker_world_positions(model)
            feet = [mw0[n][1] for n in mw0 if any(t in n for t in ("HEE", "MT", "TOE", "SOLE"))]
            if "HTOP" in mw0 and feet:
                ys = np.asarray(global_transform.apply(np.asarray(sample.verts, float)), float)[:, 1]
                g = float(np.ptp(ys) / (mw0["HTOP"][1] - min(feet)))
                for bd in ("calcn_r", "calcn_l", "toes_r", "toes_l", "hand_r", "hand_l"):
                    if bd in seg:
                        impose_body_scale(seg, model, bd, (g, g, g))
                        aniso[bd] = (g, g, g)
                print(f"  [echelle globale] stature sujet / modele = {g:.3f} (pieds, mains)")
        if foot == "sole":
            aniso.update(foot_from_sole(seg, model, sample, global_transform, _full_landmarks))
        if imposed_scales:
            for b_name, sc in imposed_scales.items():
                if b_name in {bd.name for bd in model.bodies}:
                    impose_body_scale(seg, model, b_name, sc)
                    aniso[b_name] = tuple(float(x) for x in np.asarray(sc, float).reshape(3))
        if pelvis == "hara2016":
            _longs = {k: v for k, v in (imposed_scales or {}).items()
                      if k in ("femur_r", "femur_l", "tibia_r", "tibia_l")}
            aniso.update(pelvis_hara2016(seg, model, sample, global_transform, long_scales=_longs or None))
        elif pelvis == "hara2016_3d":
            _sc, _hjc = pelvis_hara2016_3d(seg, model, sample, global_transform, _full_landmarks)
            _ctx["hips"] = _sc.pop("_hips", None)
            aniso.update(_sc)
            if _hjc is not None:
                _ctx["hjc"] = _hjc
        elif pelvis is not None:
            raise ValueError(f"unknown pelvis option {pelvis!r} (expected None, 'hara2016' or 'hara2016_3d')")
        if trunk == "after_pelvis":
            ps = aniso.get("pelvis") or (frame_scale(seg["pelvis"]),) * 3
            aniso.update(trunk_after_pelvis(seg, model, sample, global_transform, ps, target_span=trunk_span,
                                            hip_mid=_ctx.get("hjc")))
        elif trunk == "acromion":
            ps = aniso.get("pelvis") or (frame_scale(seg["pelvis"]),) * 3
            aniso.update(trunk_from_acromions(seg, model, sample, global_transform, ps,
                                              {n: v for n, v in _full_landmarks.items()}, target_span=trunk_span))
        elif trunk == "after_pelvis_acromion_width":
            # hauteur comme after_pelvis (le squelette est a la hauteur du gold), LARGEUR (z du torse) par la distance
            # entre acromions du sujet sur celle des marqueurs du modele (les centres d'epaule du rig sont trop serres)
            ps = aniso.get("pelvis") or (frame_scale(seg["pelvis"]),) * 3
            th = trunk_after_pelvis(seg, model, sample, global_transform, ps, target_span=trunk_span,
                                    hip_mid=_ctx.get("hjc"))
            aw = trunk_from_acromions({k: v.copy() for k, v in seg.items()}, model, sample, global_transform, ps,
                                      _full_landmarks)
            if th and aw:
                w = aw["torso"][2]
                for bd, sc in th.items():
                    sc2 = (sc[0], sc[1], w) if bd in ("torso", "Abdomen") else sc
                    impose_body_scale(seg, model, bd, sc2)
                    th[bd] = tuple(float(x) for x in sc2)
            aniso.update(th)
        elif trunk == "skin_height":
            ps = aniso.get("pelvis") or (frame_scale(seg["pelvis"]),) * 3
            aniso.update(trunk_from_skin_height(seg, model, sample, global_transform, ps, _full_landmarks))
        elif trunk is not None:
            raise ValueError(f"unknown trunk option {trunk!r} (expected None, 'after_pelvis', 'acromion' or 'skin_height')")
        if chain == "least_squares":
            hips = _ctx.get("hips") or {}
            res = place_least_squares(seg, model, sample, global_transform,
                                      hip_targets={f"hip_{sd}": v for sd, v in hips.items()})
            print("  [placement moindres carres] residus (mm) : "
                  + ", ".join(f"{k} {1000 * v:.1f}" for k, v in sorted(res.items())))
        elif chain:
            chain_consistent(seg, model)
        return aniso

    aniso = _ajuster(seg_transforms)
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
                _ajuster(seg_no_roll)  # memes tailles et meme chaine que les autres marqueurs
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

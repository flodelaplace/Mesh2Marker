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

from .alignment import align_mhr_to_opensim
from .linking import centroid_vertex, reposition_marker_to_vertex
from .mhr import MhrSample
from .morph import ShapeBasis, morph
from .osim import parse_osim, write_osim_with_markers
from .segment_align import compute_segment_transforms


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
) -> dict[str, tuple[float, float, float]]:
    """Write ``dst`` = ``src`` with mapped markers moved onto ``sample``'s skin.

    Pure function: parses the source model, aligns the sample to it, computes the
    per-segment transforms, and for every mapped marker present in the model with at
    least one vertex index, takes the centroid vertex and inverts the segment world
    matrix to get the new segment-local location. Returns the written locations.
    """
    index_map = _as_index_map(correspondence)
    model = parse_osim(src_osim_path)
    global_transform, _, _ = align_mhr_to_opensim(sample, model)
    seg_transforms = compute_segment_transforms(sample, model, global_transform)

    model_markers = {m.name for m in model.markers}
    locations: dict[str, tuple[float, float, float]] = {}
    for marker_name, indices in index_map.items():
        if not indices or marker_name not in model_markers:
            continue
        idx = centroid_vertex(sample.verts, list(indices))
        local = reposition_marker_to_vertex(
            model, marker_name, sample.verts, idx, global_transform, seg_transforms
        )
        locations[marker_name] = (float(local[0]), float(local[1]), float(local[2]))

    write_osim_with_markers(src_osim_path, dst_osim_path, locations)
    return locations


def sample_from_betas(basis: ShapeBasis, betas) -> MhrSample:
    """Sugar: the morphed rest-pose sample for the given shape coefficients."""
    return morph(basis, betas)

"""World positions of OpenSim model markers, optionally placed by alignment.

Pure core: stdlib + numpy only. No bpy, no pydantic.

Each OsimMarker has a ``location`` in its parent body's local frame. Its neutral
world position is ``world[parent_body]`` applied to that location (world from
:func:`mesh2marker.kinematics.forward_kinematics`). When per-segment corrections
are provided, the body's correction (a 4x4 in the OpenSim world frame, the same
chain used to place segment geometry) is applied on top. The Z-up conversion stays
on the bpy side.
"""

from __future__ import annotations

import numpy as np

from .kinematics import forward_kinematics
from .osim import OsimModel
from .procrustes import SimilarityTransform


def neutral_marker_world_positions(model: OsimModel) -> dict[str, np.ndarray]:
    """Each marker's OpenSim-neutral world position: ``world[parent] @ location``."""
    world = forward_kinematics(model)
    positions: dict[str, np.ndarray] = {}
    for marker in model.markers:
        body_world = world.get(marker.parent_body)
        if body_world is None:
            continue
        loc = np.asarray(marker.location, dtype=float)
        positions[marker.name] = body_world.rotation @ loc + body_world.translation
    return positions


def marker_world_positions(
    model: OsimModel,
    seg_transforms: dict[str, np.ndarray] | None = None,
    global_transform: SimilarityTransform | None = None,
) -> dict[str, np.ndarray]:
    """World position of every marker, optionally placed by the alignment.

    - ``seg_transforms`` given: apply the parent body's per-segment correction,
      i.e. ``seg_transforms[parent] @ (world[parent] @ location)`` -- the same chain
      as segment geometry.
    - else ``global_transform`` given: place by the global similarity alone.
    - else: neutral positions (``world[parent] @ location``).
    """
    parent_of = {marker.name: marker.parent_body for marker in model.markers}
    positions = neutral_marker_world_positions(model)

    out: dict[str, np.ndarray] = {}
    for name, point in positions.items():
        if seg_transforms is not None:
            matrix = seg_transforms.get(parent_of[name])
            if matrix is not None:
                point = matrix[:3, :3] @ point + matrix[:3, 3]
        elif global_transform is not None:
            point = np.asarray(global_transform.apply(point), dtype=float)
        out[name] = point
    return out

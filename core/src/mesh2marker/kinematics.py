"""Neutral-pose forward kinematics for a parsed OpenSim model.

Pure core: stdlib + numpy only (numpy ships inside Blender's embedded Python, so
this respects the zero-wheel rule). No pydantic, no bpy.

From an :class:`~mesh2marker.osim.OsimModel` this computes the world transform of
every body and the world positions of the joint centres in the neutral pose (all
coordinates at zero). The joint-centre cloud is the target for Procrustes
pre-alignment.

Conventions
-----------
- A PhysicalOffsetFrame ``orientation`` is a body-fixed (intrinsic) Euler XYZ
  sequence: successive rotations about X, then Y, then Z, i.e.
  ``R = Rx(a) @ Ry(b) @ Rz(c)``. An offset is a rigid transform (this rotation
  plus a translation).
- Neutral pose: the joint's own coordinate transform is taken to be the identity.
  This is exact for the standard rotational/translational joints (every
  coordinate at zero) and a very good approximation for coupled joints such as
  ``walker_knee`` (whose spline-driven translation is ~0 at full extension). It is
  sufficient for pre-alignment and for displaying the model in the neutral pose.
  Hence a child body's world transform is
  ``child_world = parent_world ∘ T(parent_offset) ∘ T(child_offset)^{-1}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .osim import OsimFrameOffset, OsimModel


@dataclass(eq=False)
class Transform:
    """A rigid transform mapping a local point to its parent frame: ``p' = R p + t``."""

    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # length 3

    @classmethod
    def identity(cls) -> Transform:
        return cls(np.eye(3), np.zeros(3))

    def compose(self, other: Transform) -> Transform:
        """Return ``self ∘ other`` (apply ``other`` first, then ``self``)."""
        return Transform(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )

    def inverse(self) -> Transform:
        rot_t = self.rotation.T
        return Transform(rot_t, -rot_t @ self.translation)


def euler_xyz_to_matrix(angles: list[float] | np.ndarray) -> np.ndarray:
    """Body-fixed intrinsic Euler XYZ angles -> 3x3 matrix ``Rx(a) @ Ry(b) @ Rz(c)``."""
    a, b, c = (float(x) for x in angles)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _offset_transform(offset: OsimFrameOffset) -> Transform:
    translation = (
        np.asarray(offset.translation, dtype=float)
        if offset.translation
        else np.zeros(3)
    )
    orientation = offset.orientation if offset.orientation else [0.0, 0.0, 0.0]
    return Transform(euler_xyz_to_matrix(orientation), translation)


def forward_kinematics(model: OsimModel) -> dict[str, Transform]:
    """World transform of every body in the neutral pose; ``ground`` is identity.

    Walks the kinematic tree from ``ground`` via :meth:`OsimModel.adjacency`.
    Raises :class:`ValueError` on a body that is the child of several joints, on a
    cycle, or if some bodies are not reachable from ``ground``.
    """
    child_to_joint = {}
    for joint in model.joints:
        if joint.child_body in child_to_joint:
            raise ValueError(
                f"body {joint.child_body!r} is the child of multiple joints"
            )
        child_to_joint[joint.child_body] = joint

    adjacency = model.adjacency()
    world: dict[str, Transform] = {"ground": Transform.identity()}

    # DFS from ground; the tree is acyclic so every body is visited once.
    stack = ["ground"]
    while stack:
        parent = stack.pop()
        for child in adjacency.get(parent, []):
            if child in world:
                raise ValueError(f"cycle or repeated body detected at {child!r}")
            joint = child_to_joint[child]
            world[child] = (
                world[parent]
                .compose(_offset_transform(joint.parent_offset))
                .compose(_offset_transform(joint.child_offset).inverse())
            )
            stack.append(child)

    unreached = {b.name for b in model.bodies} - set(world)
    if unreached:
        raise ValueError(
            f"bodies not reachable from ground (missing parent or disconnected): "
            f"{sorted(unreached)}"
        )
    return world


def joint_centers(model: OsimModel) -> dict[str, list[float]]:
    """World position of each joint's parent-offset-frame origin (neutral pose).

    Keyed by joint name; this is the target cloud for Procrustes.
    """
    world = forward_kinematics(model)
    centers: dict[str, list[float]] = {}
    for joint in model.joints:
        parent_world = world[joint.parent_body]
        frame = parent_world.compose(_offset_transform(joint.parent_offset))
        centers[joint.name] = frame.translation.tolist()
    return centers

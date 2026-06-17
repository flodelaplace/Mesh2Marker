"""Marker <-> MHR vertex links.

Pure core: stdlib + numpy only. No bpy, no pydantic, no file IO (full export is a
later ticket).

MHR topology is fixed, so a link stores vertex INDICES, not positions. When several
vertices are picked, the representative (retained) index is the one closest to their
geometric centroid -- a real vertex index. By convention the representative is kept
first in :attr:`MarkerLink.vertex_indices`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def centroid_vertex(verts: np.ndarray, indices: list[int]) -> int:
    """Index (among ``indices``) of the vertex closest to their centroid.

    A single index returns itself. Raises :class:`ValueError` if ``indices`` is
    empty.
    """
    if len(indices) == 0:
        raise ValueError("indices must not be empty")
    if len(indices) == 1:
        return int(indices[0])
    points = np.asarray(verts, dtype=float)[indices]
    centroid = points.mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    return int(indices[nearest])


def ordered_indices(verts: np.ndarray, indices: list[int]) -> list[int]:
    """Selected indices with the representative (centroid) vertex placed first."""
    idx = [int(i) for i in indices]
    if not idx:
        raise ValueError("indices must not be empty")
    if len(idx) == 1:
        return idx
    chosen = centroid_vertex(verts, idx)
    return [chosen] + [i for i in idx if i != chosen]


@dataclass
class MarkerLink:
    marker_name: str
    vertex_indices: list[int]

    @property
    def chosen_index(self) -> int:
        """The retained representative vertex index (centroid if several)."""
        if not self.vertex_indices:
            raise ValueError(f"marker {self.marker_name!r} has no vertices")
        return self.vertex_indices[0]


class LinkSet:
    """A set of marker -> vertex-index links, keyed by marker name."""

    def __init__(self) -> None:
        self._links: dict[str, MarkerLink] = {}

    def add_link(
        self,
        marker_name: str,
        indices: list[int],
        verts: np.ndarray | None = None,
    ) -> MarkerLink:
        """Add or replace the link for ``marker_name``.

        When ``verts`` is given and several indices are picked, the representative
        (centroid) vertex is placed first. Raises :class:`ValueError` if empty.
        """
        idx = [int(i) for i in indices]
        if not idx:
            raise ValueError("indices must not be empty")
        if verts is not None and len(idx) > 1:
            idx = ordered_indices(verts, idx)
        link = MarkerLink(marker_name, idx)
        self._links[marker_name] = link
        return link

    def remove_link(self, marker_name: str) -> None:
        self._links.pop(marker_name, None)

    def get(self, marker_name: str) -> MarkerLink | None:
        return self._links.get(marker_name)

    def __contains__(self, marker_name: str) -> bool:
        return marker_name in self._links

    def __len__(self) -> int:
        return len(self._links)

    def to_records(self) -> list[dict]:
        return [
            {"marker": link.marker_name, "vertex_indices": list(link.vertex_indices)}
            for link in self._links.values()
        ]

    @classmethod
    def from_records(cls, records: list[dict]) -> LinkSet:
        linkset = cls()
        for record in records:
            linkset.add_link(record["marker"], record["vertex_indices"])
        return linkset


def validate_against_known(
    linkset: LinkSet, known: dict[str, int]
) -> dict[str, bool]:
    """Compare each link's retained index to a known ground-truth index.

    For every marker present in both ``known`` and ``linkset``, returns whether the
    retained (representative) index matches the expected one.
    """
    result: dict[str, bool] = {}
    for marker_name, expected in known.items():
        link = linkset.get(marker_name)
        if link is None:
            continue
        result[marker_name] = link.chosen_index == int(expected)
    return result

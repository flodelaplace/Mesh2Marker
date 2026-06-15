"""Strong, schema-level validation with pydantic v2.

This is the ONLY module in the package that imports pydantic. It lives behind the
optional ``[validation]`` extra and is used by CI and by the Mesh2Sim pipeline
(which revalidates the file anyway). It is intentionally over-engineered relative
to :mod:`mesh2marker.checks`: structural typing, ``extra='forbid'``, and a
cross-check against the shared vocabulary.

It must NEVER be imported from the Blender add-on runtime path, and the package
``__init__`` must not import it, so that ``import mesh2marker`` /
``mesh2marker.io`` / ``mesh2marker.models`` pull no pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .models import SCHEMA_VERSION
from .vocabulary import LANDMARK_NAMES, SEGMENTS_BY_MODEL


class FrameAlignmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rotation: list[list[float]]
    translation: list[float]
    scale: float

    @field_validator("rotation")
    @classmethod
    def _rotation_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation must be a 3x3 matrix")
        return v

    @field_validator("translation")
    @classmethod
    def _translation_len3(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("translation must have length 3")
        return v


class MarkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mhr_vertices: list[int]
    opensim_body: str
    local_offset: list[float]
    fixed: bool
    synthpose_index: int | None = None

    @field_validator("name")
    @classmethod
    def _name_known(cls, v: str) -> str:
        if v not in LANDMARK_NAMES:
            raise ValueError(f"unknown marker name: {v!r}")
        return v

    @field_validator("mhr_vertices")
    @classmethod
    def _vertices_nonempty_nonneg(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("mhr_vertices must not be empty")
        if any(i < 0 for i in v):
            raise ValueError("mhr_vertices must be integers >= 0")
        return v

    @field_validator("local_offset")
    @classmethod
    def _offset_len3(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("local_offset must have length 3")
        return v


class CorrespondenceFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    mhr_topology_id: str
    opensim_model: str
    marker_set: str
    frame_alignment: FrameAlignmentModel
    markers: list[MarkerModel]

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_check(self) -> CorrespondenceFileModel:
        bodies = SEGMENTS_BY_MODEL.get(self.opensim_model)
        if bodies is None:
            raise ValueError(f"unknown opensim model: {self.opensim_model!r}")

        seen: set[str] = set()
        for m in self.markers:
            if m.name in seen:
                raise ValueError(f"duplicate marker name: {m.name!r}")
            seen.add(m.name)
            if m.opensim_body not in bodies:
                raise ValueError(
                    f"unknown opensim body for model {self.opensim_model!r}: "
                    f"{m.opensim_body!r}"
                )
        return self


def validate_strict(data: dict) -> None:
    """Validate ``data`` against the schema, raising on any invalidity."""
    CorrespondenceFileModel.model_validate(data)

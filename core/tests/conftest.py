"""Shared fixtures: a valid dummy correspondence file."""

from dataclasses import asdict

import pytest

from mesh2marker.models import (
    SCHEMA_VERSION,
    CorrespondenceFile,
    FrameAlignment,
    Marker,
)


def make_valid_corr() -> CorrespondenceFile:
    return CorrespondenceFile(
        schema_version=SCHEMA_VERSION,
        mhr_topology_id="mhr-template-v1",
        opensim_model="Pose2Sim_Wholebody",
        marker_set="pose2sim_wholebody_73",
        frame_alignment=FrameAlignment(
            rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation=[0.0, 0.0, 0.0],
            scale=1.0,
        ),
        markers=[
            Marker(
                name="RASI",
                mhr_vertices=[10],
                opensim_body="pelvis",
                local_offset=[0.0, 0.0, 0.0],
                fixed=True,
                synthpose_index=0,
            ),
            Marker(
                name="RLFC",
                mhr_vertices=[20, 21],
                opensim_body="femur_r",
                local_offset=[0.01, -0.02, 0.03],
                fixed=False,
                synthpose_index=None,
            ),
        ],
    )


@pytest.fixture
def valid_corr() -> CorrespondenceFile:
    return make_valid_corr()


@pytest.fixture
def valid_data() -> dict:
    return asdict(make_valid_corr())

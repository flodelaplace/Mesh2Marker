"""Lightweight checks: valid -> [], one failing case per rule."""

from mesh2marker import checks


def test_valid_returns_no_errors(valid_corr):
    assert checks.validate(valid_corr) == []


def test_bad_schema_version(valid_corr):
    valid_corr.schema_version = "0.9"
    errors = checks.validate(valid_corr)
    assert any("schema_version" in e for e in errors)


def test_unknown_model(valid_corr):
    valid_corr.opensim_model = "NotAModel"
    errors = checks.validate(valid_corr)
    assert any("unknown opensim model" in e for e in errors)


def test_duplicate_marker_name(valid_corr):
    valid_corr.markers[1].name = valid_corr.markers[0].name
    valid_corr.markers[1].opensim_body = valid_corr.markers[0].opensim_body
    errors = checks.validate(valid_corr)
    assert any("duplicate marker name" in e for e in errors)


def test_unknown_marker_name(valid_corr):
    valid_corr.markers[0].name = "not_a_landmark"
    errors = checks.validate(valid_corr)
    assert any("unknown marker name" in e for e in errors)


def test_unknown_body(valid_corr):
    valid_corr.markers[0].opensim_body = "not_a_body"
    errors = checks.validate(valid_corr)
    assert any("unknown opensim body" in e for e in errors)


def test_offset_wrong_length(valid_corr):
    valid_corr.markers[0].local_offset = [0.0, 0.0]
    errors = checks.validate(valid_corr)
    assert any("local_offset" in e for e in errors)


def test_empty_vertices(valid_corr):
    valid_corr.markers[0].mhr_vertices = []
    errors = checks.validate(valid_corr)
    assert any("must not be empty" in e for e in errors)


def test_negative_vertex(valid_corr):
    valid_corr.markers[0].mhr_vertices = [-1]
    errors = checks.validate(valid_corr)
    assert any("integers >= 0" in e for e in errors)


def test_rotation_not_3x3(valid_corr):
    valid_corr.frame_alignment.rotation = [[1.0, 0.0], [0.0, 1.0]]
    errors = checks.validate(valid_corr)
    assert any("rotation" in e for e in errors)


def test_translation_wrong_length(valid_corr):
    valid_corr.frame_alignment.translation = [0.0, 0.0]
    errors = checks.validate(valid_corr)
    assert any("translation" in e for e in errors)

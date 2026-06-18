"""Strong pydantic validation: valid passes, invalid raises.

pydantic lives only in the optional [validation] extra, so skip the whole module
(never error) when it is absent.
"""

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from mesh2marker import validation  # noqa: E402


def test_valid_passes(valid_data):
    validation.validate_strict(valid_data)  # must not raise


def test_unknown_name_raises(valid_data):
    valid_data["markers"][0]["name"] = "not_a_landmark"
    with pytest.raises(ValidationError):
        validation.validate_strict(valid_data)


def test_unknown_body_raises(valid_data):
    valid_data["markers"][0]["opensim_body"] = "not_a_body"
    with pytest.raises(ValidationError):
        validation.validate_strict(valid_data)


def test_bad_schema_version_raises(valid_data):
    valid_data["schema_version"] = "0.9"
    with pytest.raises(ValidationError):
        validation.validate_strict(valid_data)


def test_extra_field_forbidden(valid_data):
    valid_data["unexpected"] = True
    with pytest.raises(ValidationError):
        validation.validate_strict(valid_data)

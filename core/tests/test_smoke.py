"""Smoke test: the core package imports and exposes a version."""

import mesh2marker


def test_package_imports():
    assert mesh2marker.__version__ == "0.0.0"

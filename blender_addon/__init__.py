"""Mesh2Marker Blender add-on (thin bpy layer).

This module is the entry point Blender loads. It contains no business logic:
everything testable lives in the `mesh2marker` core package. For now it only
registers and unregisters cleanly.

Dev shim: when running from a source checkout the `mesh2marker` core is not
installed in Blender's embedded Python. If it is not importable, we add the
sibling `core/src` directory to ``sys.path`` so edits to the core are reflected
live without building or installing a wheel. In a packaged release the core is
bundled as a wheel via the manifest, so this shim does nothing.
"""

import importlib.util
import sys
from pathlib import Path


def _ensure_core_on_path() -> None:
    if importlib.util.find_spec("mesh2marker") is not None:
        return
    core_src = Path(__file__).resolve().parent.parent / "core" / "src"
    if core_src.is_dir():
        sys.path.insert(0, str(core_src))


_ensure_core_on_path()


def register() -> None:
    """Register add-on classes. Empty until the UI tickets land."""


def unregister() -> None:
    """Unregister add-on classes. Empty until the UI tickets land."""

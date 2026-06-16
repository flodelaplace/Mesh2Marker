"""Mesh2Marker Blender add-on (thin bpy layer).

This module is the entry point Blender loads. It contains NO business logic:
everything testable lives in the `mesh2marker` core package. The bpy layer only
reads files through the core and pushes the resulting arrays into Blender.

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

import bpy  # noqa: E402 (must follow the sys.path shim above)
from bpy.props import PointerProperty, StringProperty  # noqa: E402
from bpy.types import Operator, Panel, PropertyGroup  # noqa: E402

MHR_OBJECT_NAME = "MHR_body"


class Mesh2MarkerProperties(PropertyGroup):
    npz_path: StringProperty(
        name="NPZ path",
        description="Path to the MHR .npz sample to load",
        subtype="FILE_PATH",
        default="",
    )


class MESH2MARKER_OT_load_mhr(Operator):
    """Load an MHR .npz through the core and build the body mesh in the scene."""

    bl_idname = "mesh2marker.load_mhr"
    bl_label = "Load MHR mesh"
    bl_description = "Load an MHR .npz sample and build the body mesh in the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        raw_path = props.npz_path.strip() if props.npz_path else ""
        if not raw_path:
            self.report({"ERROR"}, "NPZ path is empty")
            return {"CANCELLED"}

        path = bpy.path.abspath(raw_path)

        # Core does all the parsing/validation; the bpy layer stays thin.
        from mesh2marker.mhr import load_mhr_npz

        try:
            sample = load_mhr_npz(path)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Failed to load MHR npz: {exc}")
            return {"CANCELLED"}

        # Replace any existing MHR body so repeated loads do not pile up.
        existing = bpy.data.objects.get(MHR_OBJECT_NAME)
        if existing is not None:
            old_mesh = existing.data
            bpy.data.objects.remove(existing, do_unlink=True)
            if old_mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

        mesh = bpy.data.meshes.new(MHR_OBJECT_NAME)
        mesh.from_pydata(sample.verts.tolist(), [], sample.faces.tolist())
        mesh.update()

        obj = bpy.data.objects.new(MHR_OBJECT_NAME, mesh)
        context.scene.collection.objects.link(obj)

        self.report(
            {"INFO"},
            f"Loaded MHR mesh: {len(sample.verts)} verts, {len(sample.faces)} faces "
            f"(frame {sample.frame_index}, {sample.coordinate_frame})",
        )
        return {"FINISHED"}


class MESH2MARKER_PT_panel(Panel):
    bl_label = "Mesh2Marker"
    bl_idname = "MESH2MARKER_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh2Marker"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mesh2marker
        layout.prop(props, "npz_path")
        layout.operator(MESH2MARKER_OT_load_mhr.bl_idname, icon="MESH_DATA")


_CLASSES = (
    Mesh2MarkerProperties,
    MESH2MARKER_OT_load_mhr,
    MESH2MARKER_PT_panel,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh2marker = PointerProperty(type=Mesh2MarkerProperties)


def unregister() -> None:
    del bpy.types.Scene.mesh2marker
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

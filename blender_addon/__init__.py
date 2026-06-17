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

# In a source checkout the core lives in a sibling core/src (dev mode); in a
# packaged release it is bundled as a wheel and this directory does not exist.
_CORE_SRC = Path(__file__).resolve().parent.parent / "core" / "src"
_DEV_MODE = _CORE_SRC.is_dir()


def _ensure_core_on_path() -> None:
    if importlib.util.find_spec("mesh2marker") is not None:
        return
    if _CORE_SRC.is_dir():
        sys.path.insert(0, str(_CORE_SRC))


def _reload_core() -> None:
    """In dev mode, drop cached `mesh2marker` modules so live edits take effect.

    Blender keeps imported modules in sys.modules for the whole session, so a
    core source edit would otherwise stay invisible until Blender restarts. This
    is a no-op for a packaged (wheel) install.
    """
    if not _DEV_MODE:
        return
    stale = [
        name
        for name in list(sys.modules)
        if name == "mesh2marker" or name.startswith("mesh2marker.")
    ]
    for name in stale:
        del sys.modules[name]
    importlib.invalidate_caches()


_ensure_core_on_path()

import bmesh  # noqa: E402
import bpy  # noqa: E402 (must follow the sys.path shim above)
import mathutils  # noqa: E402
from bpy.props import (  # noqa: E402
    BoolProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup  # noqa: E402

MHR_OBJECT_NAME = "MHR_body"
OPENSIM_COLLECTION_NAME = "OpenSim_model"
MARKERS_COLLECTION_NAME = "markers"
MARKER_MATERIAL = "Mesh2Marker_marker"


def _find_view3d_shading(context):
    """Return the shading settings of the first 3D viewport, or None if absent."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return None
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            if space is not None:
                return space.shading
    return None


def _update_mesh_alpha(self, context):
    """Live-apply the slider to the viewport X-ray alpha when X-ray is on."""
    shading = _find_view3d_shading(context)
    if shading is not None and shading.show_xray:
        shading.xray_alpha = self.mesh_alpha


def _marker_material():
    """Get/create the distinct (red) material for marker spheres."""
    mat = bpy.data.materials.get(MARKER_MATERIAL) or bpy.data.materials.new(
        MARKER_MATERIAL
    )
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.9, 0.05, 0.05, 1.0)
    mat.diffuse_color = (0.9, 0.05, 0.05, 1.0)  # solid-view colour
    return mat


def _marker_sphere_mesh(radius: float):
    """Build one small UV-sphere mesh datablock shared by all marker objects."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=radius)
    mesh = bpy.data.meshes.new("Mesh2Marker_marker_sphere")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(_marker_material())
    return mesh


class Mesh2MarkerProperties(PropertyGroup):
    npz_path: StringProperty(
        name="NPZ path",
        description="Path to the MHR .npz sample to load",
        subtype="FILE_PATH",
        default="",
    )
    osim_path: StringProperty(
        name="OSIM path",
        description="Path to the OpenSim .osim model file",
        subtype="FILE_PATH",
        default="",
    )
    geometry_dir: StringProperty(
        name="Geometry dir",
        description="Directory holding the segment geometry files (.stl)",
        subtype="DIR_PATH",
        default="",
    )
    align_skeleton: BoolProperty(
        name="Align skeleton to MHR mesh (per-segment)",
        description=(
            "Fit each long bone between its two joint centres onto the MHR mesh "
            "(needs the NPZ path). When off, segments are placed in the neutral pose"
        ),
        default=True,
    )
    mesh_alpha: FloatProperty(
        name="Mesh alpha",
        description="Viewport X-ray transparency, lower = more see-through",
        default=0.25,
        min=0.05,
        max=1.0,
        update=_update_mesh_alpha,
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
        _reload_core()
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


class MESH2MARKER_OT_load_opensim(Operator):
    """Parse an OpenSim model and display its segments assembled in the neutral pose."""

    bl_idname = "mesh2marker.load_opensim"
    bl_label = "Load OpenSim model"
    bl_description = (
        "Parse the .osim model, run forward kinematics, and place each segment's "
        "geometry standing (Z-up)"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        osim_path = bpy.path.abspath(props.osim_path.strip()) if props.osim_path else ""
        geometry_dir = (
            bpy.path.abspath(props.geometry_dir.strip()) if props.geometry_dir else ""
        )
        if not osim_path:
            self.report({"ERROR"}, "OSIM path is empty")
            return {"CANCELLED"}
        if not geometry_dir:
            self.report({"ERROR"}, "Geometry directory is empty")
            return {"CANCELLED"}

        # All parsing, kinematics, path resolution and matrices come from the core.
        _reload_core()
        from mesh2marker.geometry import (
            Y_UP_TO_Z_UP,
            geometry_world_matrix,
            resolve_geometry_file,
        )
        from mesh2marker.kinematics import forward_kinematics
        from mesh2marker.osim import parse_osim

        try:
            model = parse_osim(osim_path)
            world = forward_kinematics(model)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Failed to load OpenSim model: {exc}")
            return {"CANCELLED"}

        # Optional per-segment correction: fit each long bone onto the MHR mesh.
        # All matrices come from the core; the bpy layer only multiplies them.
        seg_transforms = None
        npz_path = bpy.path.abspath(props.npz_path.strip()) if props.npz_path else ""
        if props.align_skeleton and npz_path:
            from mesh2marker.alignment import align_mhr_to_opensim
            from mesh2marker.mhr import load_mhr_npz
            from mesh2marker.segment_align import compute_segment_transforms

            try:
                sample = load_mhr_npz(npz_path)
                global_transform, _, _ = align_mhr_to_opensim(sample, model)
                seg_transforms = compute_segment_transforms(
                    sample, model, global_transform
                )
            except (OSError, ValueError) as exc:
                self.report({"WARNING"}, f"Per-segment align skipped: {exc}")
                seg_transforms = None

        # Replace any previous import so reloads do not pile up duplicates.
        old = bpy.data.collections.get(OPENSIM_COLLECTION_NAME)
        if old is not None:
            for child in list(old.children):  # e.g. the markers sub-collection
                for obj in list(child.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(child)
            for obj in list(old.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(old)

        collection = bpy.data.collections.new(OPENSIM_COLLECTION_NAME)
        context.scene.collection.children.link(collection)

        conversion = mathutils.Matrix(Y_UP_TO_Z_UP.tolist())
        n_segments = 0
        for body in model.bodies:
            body_world = world.get(body.name)
            if body_world is None:
                continue
            for geom in body.geometries:
                resolved = resolve_geometry_file(geom.mesh_file, geometry_dir)
                if resolved is None:
                    continue  # body/geometry with no available file: skip silently

                placed = geometry_world_matrix(body_world, geom)
                if seg_transforms is not None:
                    # Correction acts in the OpenSim world frame, before Z-up.
                    placed = seg_transforms[body.name] @ placed
                final = conversion @ mathutils.Matrix(placed.tolist())

                bpy.ops.wm.stl_import(filepath=str(resolved))
                for obj in context.selected_objects:
                    for parent in list(obj.users_collection):
                        parent.objects.unlink(obj)
                    collection.objects.link(obj)
                    obj.matrix_world = final
                n_segments += 1

        # Marker spheres, placed by the same per-segment chain as the geometry.
        n_markers = 0
        if seg_transforms is not None:
            from mesh2marker.markers import marker_world_positions

            positions = marker_world_positions(model, seg_transforms)
            markers_coll = bpy.data.collections.new(MARKERS_COLLECTION_NAME)
            collection.children.link(markers_coll)
            sphere_mesh = _marker_sphere_mesh(0.01)
            for name, pos in positions.items():
                marker_obj = bpy.data.objects.new(f"marker_{name}", sphere_mesh)
                markers_coll.objects.link(marker_obj)
                marker_obj.matrix_world = conversion @ mathutils.Matrix.Translation(
                    (float(pos[0]), float(pos[1]), float(pos[2]))
                )
                n_markers += 1

        self.report(
            {"INFO"},
            f"Loaded OpenSim model: {n_segments} segments, {n_markers} markers",
        )
        return {"FINISHED"}


class MESH2MARKER_OT_align_mhr(Operator):
    """Procrustes pre-align the loaded MHR body onto the OpenSim model."""

    bl_idname = "mesh2marker.align_mhr"
    bl_label = "Align MHR to OpenSim"
    bl_description = (
        "Procrustes-align the MHR keypoints onto the OpenSim joint centres and "
        "place the MHR body accordingly"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        npz_path = bpy.path.abspath(props.npz_path.strip()) if props.npz_path else ""
        osim_path = bpy.path.abspath(props.osim_path.strip()) if props.osim_path else ""
        if not npz_path:
            self.report({"ERROR"}, "NPZ path is empty")
            return {"CANCELLED"}
        if not osim_path:
            self.report({"ERROR"}, "OSIM path is empty")
            return {"CANCELLED"}

        obj = bpy.data.objects.get(MHR_OBJECT_NAME)
        if obj is None:
            self.report(
                {"ERROR"}, f"{MHR_OBJECT_NAME!r} not found; load the MHR mesh first"
            )
            return {"CANCELLED"}

        # All the computation (clouds, Procrustes, matrix) comes from the core.
        _reload_core()
        from mesh2marker.alignment import align_mhr_to_opensim, similarity_to_matrix
        from mesh2marker.geometry import Y_UP_TO_Z_UP
        from mesh2marker.mhr import load_mhr_npz
        from mesh2marker.osim import parse_osim

        try:
            sample = load_mhr_npz(npz_path)
            model = parse_osim(osim_path)
            transform, residual, pairs = align_mhr_to_opensim(sample, model)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Alignment failed: {exc}")
            return {"CANCELLED"}

        # Compose the camera->OpenSim similarity, then the OpenSim Y-up -> Blender
        # Z-up conversion (same global convention as the model display).
        conversion = mathutils.Matrix(Y_UP_TO_Z_UP.tolist())
        similarity = mathutils.Matrix(similarity_to_matrix(transform).tolist())
        obj.matrix_world = conversion @ similarity

        self.report(
            {"INFO"},
            f"Aligned MHR: {len(pairs)} pairs, scale {transform.scale:.3f}, "
            f"residual {residual * 1000:.1f} mm",
        )
        return {"FINISHED"}


class MESH2MARKER_OT_toggle_transparency(Operator):
    """Toggle viewport X-ray so bones and markers show through the skin (Solid mode)."""

    bl_idname = "mesh2marker.toggle_transparency"
    bl_label = "Toggle mesh transparency"
    bl_description = (
        "Toggle viewport X-ray: see (and select) through the mesh in Solid mode. "
        "Dose it with the Mesh alpha slider"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        shading = _find_view3d_shading(context)
        if shading is None:
            self.report({"ERROR"}, "No 3D viewport found")
            return {"CANCELLED"}

        shading.show_xray = not shading.show_xray
        if shading.show_xray:
            shading.xray_alpha = context.scene.mesh2marker.mesh_alpha
            self.report({"INFO"}, "Viewport X-ray on (dose with Mesh alpha)")
        else:
            self.report({"INFO"}, "Viewport X-ray off")
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

        col = layout.column(align=True)
        col.label(text="MHR mesh")
        col.prop(props, "npz_path")
        col.operator(MESH2MARKER_OT_load_mhr.bl_idname, icon="MESH_DATA")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="OpenSim model")
        col.prop(props, "osim_path")
        col.prop(props, "geometry_dir")
        col.prop(props, "align_skeleton")
        col.operator(MESH2MARKER_OT_load_opensim.bl_idname, icon="ARMATURE_DATA")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Alignment")
        col.operator(MESH2MARKER_OT_align_mhr.bl_idname, icon="SNAP_ON")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Display")
        col.prop(props, "mesh_alpha", slider=True)
        col.operator(MESH2MARKER_OT_toggle_transparency.bl_idname, icon="XRAY")


_CLASSES = (
    Mesh2MarkerProperties,
    MESH2MARKER_OT_load_mhr,
    MESH2MARKER_OT_load_opensim,
    MESH2MARKER_OT_align_mhr,
    MESH2MARKER_OT_toggle_transparency,
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

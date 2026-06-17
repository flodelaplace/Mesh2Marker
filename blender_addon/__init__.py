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
    CollectionProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList  # noqa: E402

MHR_OBJECT_NAME = "MHR_body"
OPENSIM_COLLECTION_NAME = "OpenSim_model"
MARKERS_COLLECTION_NAME = "markers"
MARKER_MATERIAL = "Mesh2Marker_marker"
MARKER_ACTIVE_MATERIAL = "Mesh2Marker_marker_active"
MARKER_ACTIVE_SCALE = 2.2
LINKED_VERTEX_MATERIAL = "Mesh2Marker_linked_vertex"
LINKED_VERTEX_OBJECT = "linked_vertex_indicator"


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


def _marker_active_material():
    """Get/create the highlight (green) material for the active marker sphere."""
    mat = bpy.data.materials.get(MARKER_ACTIVE_MATERIAL) or bpy.data.materials.new(
        MARKER_ACTIVE_MATERIAL
    )
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.1, 0.9, 0.1, 1.0)
    mat.diffuse_color = (0.1, 0.9, 0.1, 1.0)  # solid-view colour
    return mat


def _linked_vertex_material():
    """Get/create the (cyan) material for the linked-vertex indicator."""
    mat = bpy.data.materials.get(LINKED_VERTEX_MATERIAL) or bpy.data.materials.new(
        LINKED_VERTEX_MATERIAL
    )
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.1, 0.7, 0.9, 1.0)
    mat.diffuse_color = (0.1, 0.7, 0.9, 1.0)  # solid-view colour
    return mat


def _build_sphere_mesh(name: str, radius: float):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=radius)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _marker_sphere_mesh(radius: float):
    """Build one small UV-sphere mesh datablock shared by all marker objects."""
    mesh = _build_sphere_mesh("Mesh2Marker_marker_sphere", radius)
    mesh.materials.append(_marker_material())
    return mesh


def _linked_vertex_indicator():
    """Get/create the reusable cyan indicator object for the linked skin vertex."""
    obj = bpy.data.objects.get(LINKED_VERTEX_OBJECT)
    if obj is None:
        mesh = _build_sphere_mesh("Mesh2Marker_linked_vertex_sphere", 0.008)
        mesh.materials.append(_linked_vertex_material())
        obj = bpy.data.objects.new(LINKED_VERTEX_OBJECT, mesh)
        obj.hide_select = True  # must never block picking
        coll = bpy.data.collections.get(OPENSIM_COLLECTION_NAME)
        target = coll if coll is not None else bpy.context.scene.collection
        target.objects.link(obj)
    return obj


def _active_marker_name(props) -> str:
    idx = props.active_marker_index
    if 0 <= idx < len(props.marker_names):
        return props.marker_names[idx].name
    return ""


def _find_link_item(props, marker_name):
    for i, link in enumerate(props.links):
        if link.marker_name == marker_name:
            return i, link
    return -1, None


def _set_marker_object_material(obj, mat) -> None:
    """Assign a per-OBJECT material to a marker sphere (spheres share one mesh)."""
    if not obj.material_slots:
        return
    slot = obj.material_slots[0]
    slot.link = "OBJECT"
    slot.material = mat


def highlight_active_marker(context) -> None:
    """Colour the active marker sphere green and enlarged; the rest red at scale 1.

    No-op when the marker spheres do not exist yet.
    """
    props = context.scene.mesh2marker
    active_obj_name = ""
    active_name = _active_marker_name(props)
    if active_name:
        active_obj_name = f"marker_{active_name}"

    for obj in bpy.data.objects:
        if not obj.name.startswith("marker_"):
            continue
        if obj.name == active_obj_name:
            _set_marker_object_material(obj, _marker_active_material())
            obj.scale = (MARKER_ACTIVE_SCALE,) * 3
        else:
            _set_marker_object_material(obj, _marker_material())
            obj.scale = (1.0, 1.0, 1.0)

    update_linked_vertex_indicator(context)


def update_linked_vertex_indicator(context) -> None:
    """Show a cyan dot on the MHR vertex linked to the active marker (skin side).

    Reads the position from the displayed MHR_body mesh, so the dot follows the
    visible vertex under any alignment. Hidden when there is no link, MHR_body is
    missing, the index is out of range, or the option is off.
    """
    props = context.scene.mesh2marker
    indicator = bpy.data.objects.get(LINKED_VERTEX_OBJECT)

    idx = -1
    if getattr(props, "show_linked_vertex", True):
        active_name = _active_marker_name(props)
        if active_name:
            _, link = _find_link_item(props, active_name)
            if link is not None and link.vertex_indices:
                first = link.vertex_indices.split(",")[0]
                if first:
                    idx = int(first)

    mhr = bpy.data.objects.get(MHR_OBJECT_NAME)
    valid = (
        idx >= 0
        and mhr is not None
        and mhr.type == "MESH"
        and idx < len(mhr.data.vertices)
    )
    if not valid:
        if indicator is not None:
            indicator.hide_viewport = True
        return

    indicator = _linked_vertex_indicator()
    world_co = mhr.matrix_world @ mhr.data.vertices[idx].co
    indicator.matrix_world = mathutils.Matrix.Translation(world_co)
    indicator.hide_viewport = False
    indicator.hide_select = True


def _update_active_marker(self, context):
    highlight_active_marker(context)


def _update_show_linked_vertex(self, context):
    update_linked_vertex_indicator(context)


def _compute_alignment(props):
    """Load sample + model and compute global + per-segment transforms (all core)."""
    npz_path = bpy.path.abspath(props.npz_path.strip()) if props.npz_path else ""
    osim_path = bpy.path.abspath(props.osim_path.strip()) if props.osim_path else ""
    if not npz_path or not osim_path:
        raise ValueError("NPZ and OSIM paths must both be set")

    from mesh2marker.alignment import align_mhr_to_opensim
    from mesh2marker.mhr import load_mhr_npz
    from mesh2marker.osim import parse_osim
    from mesh2marker.segment_align import compute_segment_transforms

    sample = load_mhr_npz(npz_path)
    model = parse_osim(osim_path)
    global_transform, _, _ = align_mhr_to_opensim(sample, model)
    seg_transforms = compute_segment_transforms(sample, model, global_transform)
    return sample, model, global_transform, seg_transforms


def _vec_csv(vec) -> str:
    return f"{float(vec[0])},{float(vec[1])},{float(vec[2])}"


def _move_marker_sphere(marker_name: str, world_pos, conversion) -> None:
    """Move a marker sphere to a world position (Z-up converted), if it exists."""
    obj = bpy.data.objects.get(f"marker_{marker_name}")
    if obj is None:
        return
    obj.matrix_world = conversion @ mathutils.Matrix.Translation(
        (float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
    )


def _set_model_selectable(selectable: bool) -> None:
    """Toggle hide_select on the OpenSim model bones and marker spheres (not hidden)."""
    coll = bpy.data.collections.get(OPENSIM_COLLECTION_NAME)
    if coll is None:
        return
    hide = not selectable
    for obj in coll.objects:
        if obj.name == LINKED_VERTEX_OBJECT:
            continue  # the indicator stays unselectable regardless
        obj.hide_select = hide
    for child in coll.children:  # the markers sub-collection
        for obj in child.objects:
            obj.hide_select = hide


class MarkerNameItem(PropertyGroup):
    """One OpenSim marker name (the pickable list, filled at model load)."""

    name: StringProperty(default="")


class MarkerLinkItem(PropertyGroup):
    """A marker -> MHR vertex-index link; indices stored as CSV (centroid first).

    ``local_offset`` is the repositioned segment-local marker position (CSV "x,y,z"),
    empty until the marker is snapped to its linked vertex.
    """

    marker_name: StringProperty(default="")
    vertex_indices: StringProperty(default="")
    local_offset: StringProperty(default="")


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
    marker_names: CollectionProperty(type=MarkerNameItem)
    active_marker_index: IntProperty(default=0, update=_update_active_marker)
    links: CollectionProperty(type=MarkerLinkItem)
    show_linked_vertex: BoolProperty(
        name="Show linked vertex",
        description="Show a cyan dot on the MHR vertex linked to the active marker",
        default=True,
        update=_update_show_linked_vertex,
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

        # Store the model's marker names as the pickable list for the UIList.
        props.marker_names.clear()
        for marker in model.markers:
            props.marker_names.add().name = marker.name
        props.active_marker_index = 0

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
                # Per-object material so each sphere can be coloured independently.
                _set_marker_object_material(marker_obj, _marker_material())
                n_markers += 1

        # Highlight the initially active marker.
        highlight_active_marker(context)

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


class MESH2MARKER_UL_markers(UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        props = context.scene.mesh2marker
        _, link = _find_link_item(props, item.name)
        row = layout.row(align=True)
        if link is not None and link.vertex_indices:
            row.label(text=item.name, icon="CHECKMARK")
            row.label(text=link.vertex_indices)
        else:
            row.label(text=item.name, icon="DOT")


class MESH2MARKER_OT_link_vertices(Operator):
    """Link the selected MHR_body vertices to the active marker (centroid retained)."""

    bl_idname = "mesh2marker.link_vertices"
    bl_label = "Link selected vertices to marker"
    bl_description = "Link the selected MHR_body vertices to the active marker"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        marker_name = _active_marker_name(props)
        if not marker_name:
            self.report({"ERROR"}, "No active marker (load a model and pick one)")
            return {"CANCELLED"}

        obj = bpy.data.objects.get(MHR_OBJECT_NAME)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, f"{MHR_OBJECT_NAME!r} not found")
            return {"CANCELLED"}
        if obj.mode != "EDIT":
            self.report({"ERROR"}, "Enter Edit Mode on MHR_body and select vertices")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        selected = [v.index for v in bm.verts if v.select]
        if not selected:
            self.report({"ERROR"}, "No vertices selected")
            return {"CANCELLED"}
        coords = [None] * len(bm.verts)
        for vert in bm.verts:
            coords[vert.index] = (vert.co.x, vert.co.y, vert.co.z)

        # The centroid choice and ordering come from the core.
        _reload_core()
        from mesh2marker.linking import ordered_indices

        ordered = ordered_indices(coords, selected)
        csv = ",".join(str(i) for i in ordered)

        _, item = _find_link_item(props, marker_name)
        if item is None:
            item = props.links.add()
            item.marker_name = marker_name
        item.vertex_indices = csv

        update_linked_vertex_indicator(context)
        self.report({"INFO"}, f"Linked {marker_name}: {len(ordered)} vertex(es)")
        return {"FINISHED"}


class MESH2MARKER_OT_unlink_marker(Operator):
    """Remove the link of the active marker."""

    bl_idname = "mesh2marker.unlink_marker"
    bl_label = "Unlink marker"
    bl_description = "Remove the vertex link of the active marker"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        marker_name = _active_marker_name(props)
        if not marker_name:
            self.report({"ERROR"}, "No active marker")
            return {"CANCELLED"}
        idx, item = _find_link_item(props, marker_name)
        if item is None:
            self.report({"WARNING"}, f"{marker_name} is not linked")
            return {"CANCELLED"}
        props.links.remove(idx)
        update_linked_vertex_indicator(context)
        self.report({"INFO"}, f"Unlinked {marker_name}")
        return {"FINISHED"}


class MESH2MARKER_OT_select_linked(Operator):
    """Re-select the MHR_body vertices linked to the active marker (Edit Mode)."""

    bl_idname = "mesh2marker.select_linked"
    bl_label = "Select linked vertices"
    bl_description = "Select the vertices linked to the active marker"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.mesh2marker
        marker_name = _active_marker_name(props)
        if not marker_name:
            self.report({"ERROR"}, "No active marker")
            return {"CANCELLED"}
        _, item = _find_link_item(props, marker_name)
        if item is None or not item.vertex_indices:
            self.report({"WARNING"}, f"{marker_name} is not linked")
            return {"CANCELLED"}

        obj = bpy.data.objects.get(MHR_OBJECT_NAME)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, f"{MHR_OBJECT_NAME!r} not found")
            return {"CANCELLED"}
        if obj.mode != "EDIT":
            self.report({"ERROR"}, "Enter Edit Mode on MHR_body first")
            return {"CANCELLED"}

        indices = [int(s) for s in item.vertex_indices.split(",") if s]
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for vert in bm.verts:
            vert.select = False
        n = len(bm.verts)
        for i in indices:
            if 0 <= i < n:
                bm.verts[i].select = True
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        self.report({"INFO"}, f"Selected {len(indices)} vertex(es) for {marker_name}")
        return {"FINISHED"}


class MESH2MARKER_OT_snap_marker(Operator):
    """Reposition the active marker onto its linked MHR skin vertex."""

    bl_idname = "mesh2marker.snap_marker"
    bl_label = "Snap marker to linked vertex"
    bl_description = "Reposition the active marker onto its linked vertex (skin)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        name = _active_marker_name(props)
        if not name:
            self.report({"ERROR"}, "No active marker")
            return {"CANCELLED"}
        _, item = _find_link_item(props, name)
        if item is None or not item.vertex_indices:
            self.report({"WARNING"}, f"{name} is not linked")
            return {"CANCELLED"}

        _reload_core()
        from mesh2marker.geometry import Y_UP_TO_Z_UP
        from mesh2marker.linking import (
            reposition_marker_to_vertex,
            vertex_world_position,
        )

        try:
            sample, model, gt, seg = _compute_alignment(props)
            idx = int(item.vertex_indices.split(",")[0])
            local = reposition_marker_to_vertex(model, name, sample.verts, idx, gt, seg)
            world_pos = vertex_world_position(sample.verts, idx, gt)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Snap failed: {exc}")
            return {"CANCELLED"}

        item.local_offset = _vec_csv(local)
        _move_marker_sphere(name, world_pos, mathutils.Matrix(Y_UP_TO_Z_UP.tolist()))
        highlight_active_marker(context)
        self.report({"INFO"}, f"Snapped {name} -> local ({item.local_offset})")
        return {"FINISHED"}


class MESH2MARKER_OT_set_marker_here(Operator):
    """Link the selected MHR vertex to the active marker and snap it there."""

    bl_idname = "mesh2marker.set_marker_here"
    bl_label = "Set marker here from selected vertex"
    bl_description = "Link the selected vertex (centroid) to the active marker and snap it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        name = _active_marker_name(props)
        if not name:
            self.report({"ERROR"}, "No active marker")
            return {"CANCELLED"}
        obj = bpy.data.objects.get(MHR_OBJECT_NAME)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, f"{MHR_OBJECT_NAME!r} not found")
            return {"CANCELLED"}
        if obj.mode != "EDIT":
            self.report({"ERROR"}, "Enter Edit Mode on MHR_body and select vertices")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        selected = [v.index for v in bm.verts if v.select]
        if not selected:
            self.report({"ERROR"}, "No vertices selected")
            return {"CANCELLED"}
        coords = [None] * len(bm.verts)
        for vert in bm.verts:
            coords[vert.index] = (vert.co.x, vert.co.y, vert.co.z)

        _reload_core()
        from mesh2marker.geometry import Y_UP_TO_Z_UP
        from mesh2marker.linking import (
            ordered_indices,
            reposition_marker_to_vertex,
            vertex_world_position,
        )

        ordered = ordered_indices(coords, selected)
        try:
            sample, model, gt, seg = _compute_alignment(props)
            idx = ordered[0]
            local = reposition_marker_to_vertex(model, name, sample.verts, idx, gt, seg)
            world_pos = vertex_world_position(sample.verts, idx, gt)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Set marker failed: {exc}")
            return {"CANCELLED"}

        _, item = _find_link_item(props, name)
        if item is None:
            item = props.links.add()
            item.marker_name = name
        item.vertex_indices = ",".join(str(i) for i in ordered)
        item.local_offset = _vec_csv(local)
        _move_marker_sphere(name, world_pos, mathutils.Matrix(Y_UP_TO_Z_UP.tolist()))
        highlight_active_marker(context)
        self.report({"INFO"}, f"Set {name} at vertex {idx}")
        return {"FINISHED"}


class MESH2MARKER_OT_snap_all_markers(Operator):
    """Reposition every linked marker onto its skin vertex."""

    bl_idname = "mesh2marker.snap_all_markers"
    bl_label = "Snap ALL linked markers to skin"
    bl_description = "Reposition every linked marker onto its linked vertex"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mesh2marker
        _reload_core()
        from mesh2marker.geometry import Y_UP_TO_Z_UP
        from mesh2marker.linking import (
            reposition_marker_to_vertex,
            vertex_world_position,
        )

        try:
            sample, model, gt, seg = _compute_alignment(props)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Snap-all failed: {exc}")
            return {"CANCELLED"}

        conversion = mathutils.Matrix(Y_UP_TO_Z_UP.tolist())
        n = 0
        for item in props.links:
            if not item.vertex_indices:
                continue
            idx = int(item.vertex_indices.split(",")[0])
            try:
                local = reposition_marker_to_vertex(
                    model, item.marker_name, sample.verts, idx, gt, seg
                )
                world_pos = vertex_world_position(sample.verts, idx, gt)
            except ValueError:
                continue  # unknown marker / out-of-range index
            item.local_offset = _vec_csv(local)
            _move_marker_sphere(item.marker_name, world_pos, conversion)
            n += 1

        highlight_active_marker(context)
        self.report({"INFO"}, f"Snapped {n} markers to skin")
        return {"FINISHED"}


class MESH2MARKER_OT_auto_link(Operator):
    """Auto-link every still-unlinked marker to its nearest MHR skin vertex."""

    bl_idname = "mesh2marker.auto_link"
    bl_label = "Auto-link all markers"
    bl_description = (
        "Propose the nearest mesh vertex for each marker that has no link yet "
        "(existing links are kept)"
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

        # All computation comes from the core.
        _reload_core()
        from mesh2marker.alignment import align_mhr_to_opensim
        from mesh2marker.linking import auto_link_markers
        from mesh2marker.mhr import load_mhr_npz
        from mesh2marker.osim import parse_osim
        from mesh2marker.segment_align import compute_segment_transforms

        try:
            sample = load_mhr_npz(npz_path)
            model = parse_osim(osim_path)
            global_transform, _, _ = align_mhr_to_opensim(sample, model)
            seg_transforms = compute_segment_transforms(
                sample, model, global_transform
            )
            proposed = auto_link_markers(
                model, sample.verts, global_transform, seg_transforms
            )
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Auto-link failed: {exc}")
            return {"CANCELLED"}

        new_count = 0
        kept = 0
        for marker_name, vertex_index in proposed.items():
            _, item = _find_link_item(props, marker_name)
            if item is not None:  # preserve manual refinement / prior links
                kept += 1
                continue
            item = props.links.add()
            item.marker_name = marker_name
            item.vertex_indices = str(int(vertex_index))
            new_count += 1

        highlight_active_marker(context)
        self.report(
            {"INFO"}, f"Auto-linked {new_count} markers, kept {kept} existing"
        )
        return {"FINISHED"}


class MESH2MARKER_OT_enter_picking(Operator):
    """Edit MHR_body in vertex mode with X-ray; make the skeleton unselectable."""

    bl_idname = "mesh2marker.enter_picking"
    bl_label = "Enter picking mode"
    bl_description = (
        "Edit MHR_body in vertex/X-ray mode and lock the skeleton so clicks land on "
        "the mesh"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = bpy.data.objects.get(MHR_OBJECT_NAME)
        if obj is None or obj.type != "MESH":
            self.report(
                {"ERROR"}, f"{MHR_OBJECT_NAME!r} not found; load the MHR mesh first"
            )
            return {"CANCELLED"}

        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Skeleton and marker spheres become unselectable (kept visible).
        _set_model_selectable(False)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")

        shading = _find_view3d_shading(context)
        if shading is not None:
            shading.show_xray = True
            shading.xray_alpha = context.scene.mesh2marker.mesh_alpha

        self.report(
            {"INFO"},
            "Picking mode: select vertices on MHR_body, then Link selected vertices "
            "to marker",
        )
        return {"FINISHED"}


class MESH2MARKER_OT_exit_picking(Operator):
    """Return to Object Mode and make the skeleton selectable again."""

    bl_idname = "mesh2marker.exit_picking"
    bl_label = "Exit picking mode"
    bl_description = "Back to Object Mode; make the skeleton and markers selectable again"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _set_model_selectable(True)
        self.report({"INFO"}, "Exited picking mode")
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
        col.label(text="Markers")
        col.template_list(
            "MESH2MARKER_UL_markers",
            "",
            props,
            "marker_names",
            props,
            "active_marker_index",
            rows=6,
        )
        col.operator(MESH2MARKER_OT_auto_link.bl_idname, icon="FILE_REFRESH")
        row = col.row(align=True)
        row.operator(MESH2MARKER_OT_enter_picking.bl_idname, icon="EDITMODE_HLT")
        row.operator(MESH2MARKER_OT_exit_picking.bl_idname, icon="OBJECT_DATAMODE")
        row = col.row(align=True)
        row.operator(MESH2MARKER_OT_link_vertices.bl_idname, icon="LINKED")
        row.operator(MESH2MARKER_OT_unlink_marker.bl_idname, icon="UNLINKED")
        col.operator(
            MESH2MARKER_OT_select_linked.bl_idname, icon="RESTRICT_SELECT_OFF"
        )
        col.operator(MESH2MARKER_OT_snap_marker.bl_idname, icon="SNAP_ON")
        col.operator(MESH2MARKER_OT_set_marker_here.bl_idname, icon="VERTEXSEL")
        col.operator(MESH2MARKER_OT_snap_all_markers.bl_idname, icon="SNAP_VERTEX")
        col.prop(props, "show_linked_vertex")

        col.separator()
        col.prop(props, "mesh_alpha", slider=True)
        col.operator(MESH2MARKER_OT_toggle_transparency.bl_idname, icon="XRAY")


_CLASSES = (
    MarkerNameItem,
    MarkerLinkItem,
    Mesh2MarkerProperties,
    MESH2MARKER_OT_load_mhr,
    MESH2MARKER_OT_load_opensim,
    MESH2MARKER_OT_align_mhr,
    MESH2MARKER_OT_toggle_transparency,
    MESH2MARKER_UL_markers,
    MESH2MARKER_OT_auto_link,
    MESH2MARKER_OT_enter_picking,
    MESH2MARKER_OT_exit_picking,
    MESH2MARKER_OT_link_vertices,
    MESH2MARKER_OT_unlink_marker,
    MESH2MARKER_OT_select_linked,
    MESH2MARKER_OT_snap_marker,
    MESH2MARKER_OT_set_marker_here,
    MESH2MARKER_OT_snap_all_markers,
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

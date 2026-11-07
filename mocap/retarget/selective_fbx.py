"""Private Blender 4.5 adapter; retain native rest data and body-only animation.

Blender's All Bones switch controls key retention, not which bones are sampled.
Use scoped, shape-checked adapters for animation sampling and static bone defaults.
The installed Blender exporter remains unmodified and owns FBX transforms/writing.
Runtime-derived functions retain Blender's GPL license, not the app MIT grant.
"""
import contextlib, inspect, hashlib
from io_scene_fbx import export_fbx_bin


def body_dependencies(kind, arm):
    # PMX deformation helpers transfer the same body motion into weighted skin.
    # They are anatomical helpers, not skirt/hair/ornament animation. Omitting
    # them breaks skin when FBX drops Blender constraints.
    if kind != 'pmx':
        return set()
    stems = ('足D', 'ひざD', '足首D', '足首補助', 'ひざ補助',
             '肩補助', '手首補助', 'ひじ補助', '腕補助', '胸補助')
    return {stem+side for stem in stems for side in ('.L', '.R')
            if stem+side in arm.data.bones}


@contextlib.contextmanager
def body_only_sampler(arm, names):
    original = export_fbx_bin.fbx_animations_do
    original_objects = export_fbx_bin.fbx_data_object_elements
    source = inspect.getsource(original)
    marker = '    for ob_obj in objects:\n        if ob_obj.parented_to_armature:'
    assert source.count(marker) == 1, 'Unsupported Blender FBX sampler; review adapter'
    modified = source.replace(marker,
        '    for ob_obj in objects:\n'
        '        if not _body_export_allowed(ob_obj):\n'
        '            continue\n'
        '        if ob_obj.parented_to_armature:')
    # Match the previous per-action exporter: take time starts at zero, so
    # Blender's ordinary +1 import offset places the first sample on frame 1.
    modified = modified.replace('    bake_step = scene_data.settings.bake_anim_step',
        '    start_zero = True\n    bake_step = scene_data.settings.bake_anim_step', 1)
    sampled = set()
    def allowed(ob):
        keep = ob.is_bone and ob.bdata.id_data == arm.data and ob.bdata.name in names
        if keep: sampled.add(ob.bdata.name)
        return keep
    scope = dict(original.__globals__, _body_export_allowed=allowed)
    exec(compile(modified, '<private-body-only-fbx-sampler>', 'exec'), scope)
    # FBX Model defaults are also the skeleton's rest pose when there is no
    # skinned mesh/BindPose. Blender's exporter otherwise writes the currently
    # evaluated pose here, so skeleton-only exports import frame 1 as their rest.
    # Use Bone.matrix_local for static bone defaults only. Animation sampling and
    # the exporter's native skin bind matrices continue to use their own paths.
    object_source = inspect.getsource(original_objects)
    object_marker = 'loc, rot, scale, matrix, matrix_rot = ob_obj.fbx_object_tx(scene_data)'
    assert object_source.count(object_marker) == 1, 'Unsupported Blender FBX object writer; review adapter'
    object_modified = object_source.replace(object_marker,
        'loc, rot, scale, matrix, matrix_rot = ob_obj.fbx_object_tx(scene_data, rest=ob_obj.is_bone)')
    object_scope = dict(original_objects.__globals__)
    exec(compile(object_modified, '<private-rest-bone-fbx-writer>', 'exec'), object_scope)
    export_fbx_bin.fbx_animations_do = scope['fbx_animations_do']
    export_fbx_bin.fbx_data_object_elements = object_scope['fbx_data_object_elements']
    receipt = dict(upstream_sampler_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   upstream_object_writer_sha256=hashlib.sha256(object_source.encode()).hexdigest(), sampled=sampled)
    try:
        yield receipt
    finally:
        export_fbx_bin.fbx_animations_do = original
        export_fbx_bin.fbx_data_object_elements = original_objects

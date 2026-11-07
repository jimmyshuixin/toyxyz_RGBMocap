"""Bake only supported facial shape keys; body actions and native rigs stay intact."""
import json
from pathlib import Path
import numpy as np
import bpy
from .file_paths import io_path, save_blend
from .face_track import load_track, sample_track, map_shape_channels, mapped_source_channels


def apply_face(folder, track_path, start_seconds, frames, fps, *, rest=False, optional=False):
    from .blender_bake import curves
    folder = Path(folder)
    kind = json.loads(Path(io_path(folder / 'import.json')).read_text(encoding='utf-8'))['kind']
    source = folder / (f'{kind}-rest.blend' if rest else 'animation.blend')
    bpy.ops.wm.open_mainfile(filepath=io_path(source), load_ui=False)
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
    track = load_track(io_path(track_path))
    weights = sample_track(track, start_seconds, frames, fps)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.data.shape_keys]
    available = {key.name for obj in meshes for key in obj.data.shape_keys.key_blocks[1:]}
    channels = map_shape_channels(track['morph_names'], weights, available)
    if not channels:
        if optional:
            from mocap.storage import atomic_json
            message='This character has no supported facial shape keys; keeping its body animation. Reimport older PMX characters to include morphs.'
            print('CHARACTER: '+message, flush=True)
            atomic_json(Path(io_path(folder/'face-skipped.json')), dict(status='skipped',message=message))
            info=json.loads(Path(io_path(folder/'bake.json')).read_text(encoding='utf8'))
            return arm, info['keyed_bones'], []
        raise ValueError('This character has no supported facial shape keys. Reimport older PMX characters with morphs, or use a model with MMD/ARKit facial shapes.')
    covered = mapped_source_channels(track['morph_names'], available)
    missing = [name for name, found in zip(track['morph_names'], covered) if not found and name != '_neutral']
    if missing:
        print('CHARACTER: Facial channels unavailable on this character: ' + ', '.join(missing), flush=True)
    if (start_seconds + (frames - 1) / fps < track['start_seconds']
            or start_seconds > track['start_seconds'] + (len(track['weights']) - 1) / track['fps']):
        print('CHARACTER: This take does not overlap the face capture; facial weights stay neutral.', flush=True)
    # Manual replacement operates on a copied animation generation. Remove only
    # channels previously written by this feature which the new mapping no
    # longer uses (for example old ワ when native mapping now selects にやり).
    # Other artist-authored shape animation and every body channel are retained.
    previous = Path(io_path(folder/'face-applied.npz'))
    retired = set(load_track(previous)['morph_names'])-set(channels) if not rest and previous.is_file() else set()
    done = set()
    shape_sets = []
    for obj in meshes:
        shapes = obj.data.shape_keys
        if shapes.as_pointer() in done:
            continue
        done.add(shapes.as_pointer())
        shape_sets.append(shapes)
        if not shapes.use_relative:
            raise ValueError('Absolute shape-key characters are not supported for face capture.')
        for key in shapes.key_blocks[1:]:
            if key.name not in channels and key.name not in retired:
                continue
            path = key.path_from_id('value')
            if shapes.animation_data and any(d.data_path == path for d in shapes.animation_data.drivers):
                raise ValueError('A facial shape key is controlled by a driver: ' + key.name)

    for shapes in shape_sets:
        for key in shapes.key_blocks[1:]:
            path = key.path_from_id('value')
            if key.name in retired:
                action = shapes.animation_data.action if shapes.animation_data else None
                if action is not None:
                    collection = curves(action)
                    for curve in list(collection):
                        if curve.data_path == path and curve.array_index == 0:
                            collection.remove(curve)
                key.value = 0.
            if key.name not in channels:
                continue
            key.value = float(channels[key.name][0])
            key.keyframe_insert(data_path='value', frame=1)
            curve = next(c for c in curves(shapes.animation_data.action) if c.data_path == path)
            curve.keyframe_points.clear()
            curve.keyframe_points.add(frames)
            curve.keyframe_points.foreach_set('co', np.column_stack((np.arange(1, frames + 1), channels[key.name])).ravel())
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
            curve.update()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_set(1)
    names = sorted(channels)
    np.savez(io_path(folder / 'face-applied.npz'), morph_names=np.asarray(names),
             weights=np.column_stack([channels[name] for name in names]).astype(np.float32),
             fps=float(fps), start_seconds=float(start_seconds), detected=np.ones(frames, bool))
    save_blend(folder / 'animation.blend')
    info_path = Path(io_path(folder / 'bake.json'))
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else dict(keyed_bones=[], object_animation_channels=0)
    info.update(frames=frames, face_channels=names, face_missing_channels=missing)
    info_path.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    return arm, info['keyed_bones'], names


def preview_morphs(folder, meshes, vertex_count):
    """Sparse vertex deltas, in the same world/app coordinates as rest vertices."""
    from .mapping import A
    applied = Path(io_path(Path(folder) / 'face-applied.npz'))
    if not applied.is_file():
        return {}, None
    track = load_track(applied)
    names = track['morph_names']
    channels = {name: i for i, name in enumerate(names)}
    ids, values = [], []
    offset = 0
    for obj in meshes:
        count = len(obj.data.vertices)
        keys = obj.data.shape_keys
        if keys:
            matrix = np.asarray(obj.matrix_world)[:3, :3]
            for key in keys.key_blocks[1:]:
                if key.name not in channels:
                    continue
                if key.vertex_group:
                    raise ValueError('Vertex-group masked facial shape keys are not supported: ' + key.name)
                before = np.empty(count * 3, np.float32)
                after = np.empty(count * 3, np.float32)
                key.relative_key.data.foreach_get('co', before)
                key.data.foreach_get('co', after)
                delta = ((after - before).reshape(-1, 3) @ matrix.T @ A).astype(np.float32)
                nonzero = np.flatnonzero(np.any(delta != 0, axis=1))
                ids.append(nonzero + offset)
                values.append(np.column_stack((np.full(len(nonzero), channels[key.name]), delta[nonzero])))
        offset += count
    if sum(len(x) for x in ids) > 4_000_000:
        raise ValueError('Character facial preview exceeds the sparse morph limit.')
    vertex_ids = np.concatenate(ids) if ids else np.empty(0, np.int64)
    entries = np.concatenate(values).astype(np.float32) if values else np.empty((0, 4), np.float32)
    order = np.argsort(vertex_ids, kind='stable')
    entries = entries[order]
    counts = np.bincount(vertex_ids, minlength=vertex_count).astype(np.uint32)
    starts = np.cumsum(counts, dtype=np.uint32) - counts
    return dict(morph_rows=np.column_stack((starts, counts)), morph_entries=entries,
                morph_count=np.asarray(len(names), np.int32)), track['weights']

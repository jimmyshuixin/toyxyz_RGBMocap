import bpy,sys,json,time,numpy as np
from pathlib import Path
from mathutils import Matrix
from .file_paths import io_path, save_blend
from .mapping import real_bone_mapping
A=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
def curves(action):
    return action.fcurves if hasattr(action, 'fcurves') else next((b for l in action.layers for s in l.strips for b in s.channelbags)).fcurves

def key_values(obj, path, values):
    n = len(values)
    for index in range(values.shape[1]):
        f = next((c for c in curves(obj.animation_data.action) if c.data_path == path and c.array_index == index))
        f.keyframe_points.clear()
        f.keyframe_points.add(n)
        f.keyframe_points.foreach_set('co', np.column_stack((np.arange(1, n + 1), values[:, index])).reshape(-1))
        for k in f.keyframe_points:
            k.interpolation = 'LINEAR'
        f.update()

def rotation(M):
    (U, _, V) = np.linalg.svd(M)
    R = U @ V
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ V
    return R

def align(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1, 1))
    if c < -0.999999:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / max(1 + c, 1e-10)


def validate_pmx_ancestor_controls(arm, keyed):
    """Reject unsupported animated control parents before publishing a bake.

    Native waist-cancel controls have an explicit bake/VMD implementation.
    Other append-driven parents need an evaluated control-chain solve; treating
    them as static can preserve joint positions while corrupting orientations.
    """
    intermediates = set()
    for name in keyed:
        bone = arm.data.bones[name].parent
        while bone is not None:
            intermediates.add(bone.name)
            bone = bone.parent
    # Append rotations use the source's local pose. Ordinary shoulder-cancel
    # bones driven by an unkeyed, static shoulder-P control remain identity.
    # A transitive append source driven by a keyed elbow is different, even
    # if that intermediate source is itself unkeyed.
    grant_sources = {}
    for pb in arm.pose.bones:
        source = getattr(getattr(pb, 'mmd_bone', None), 'additional_transform_bone', '')
        if source and any(c.name in {'mmd_additional_rotation', 'mmd_additional_location'}
                          and not c.mute and c.influence > 0 for c in pb.constraints):
            grant_sources[pb.name] = source
    animated = set(keyed)
    # Independent arm IK can drive an unkeyed duplicate elbow, whose result
    # is then granted back into the mapped wrist's parent chain.
    for pb in arm.pose.bones:
        if pb.name in keyed or pb.name.startswith('腰キャンセル'):
            continue
        for constraint in pb.constraints:
            if (getattr(constraint, 'type', '') != 'IK' or constraint.mute
                    or constraint.influence <= 0):
                continue
            current = pb.bone
            count = 0
            while current is not None and (constraint.chain_count == 0 or count < constraint.chain_count):
                animated.add(current.name)
                current = current.parent
                count += 1
    while True:
        updated = animated | {name for name, source in grant_sources.items() if source in animated}
        if updated == animated:
            break
        animated = updated
    unsupported = []
    for name in sorted(intermediates - set(keyed)):
        if name.startswith('腰キャンセル'):
            continue
        if name in animated and name in grant_sources:
            unsupported.append(name)
    if unsupported:
        raise ValueError('PMX retargeting does not support active append-transform ancestors: '
                         + ', '.join(unsupported)
                         + '. Use the standard PMX variant without these auxiliary controls. '
                           'The original model has not been changed.')

def bake(kind, label, method):
    start = time.perf_counter()
    timings = {}
    bpy.ops.wm.open_mainfile(filepath=io_path(OUT / (kind + '-rest.blend')))
    rig = json.loads(Path(io_path(OUT / (kind + '-rig.json'))).read_text(encoding='utf8'))
    arm = bpy.data.objects[rig['armature']]
    arm.animation_data_clear()
    with np.load(io_path(OUT / (label + '-' + kind + '-' + method + '.npz'))) as data:
        # NPZ does not cache members: reuse these arrays across all body/finger bones.
        motion_root, motion_names, motion_world = data['root'], data['names'], data['world']
    n = len(motion_root)
    mapped = real_bone_mapping(motion_names.tolist(), rig.get('body_profile'))
    bones = list(arm.data.bones)
    indices = {b.name: i for (i, b) in enumerate(bones)}
    missing = sorted(set(mapped) - set(indices))
    if missing:
        raise ValueError('Character bones changed after import: ' + ', '.join(missing) + '. Import the character again.')
    rest = np.array([arm.matrix_world @ b.head_local for b in bones])
    R0 = np.array([rotation(np.array(arm.matrix_world @ b.matrix_local)[:3, :3]) for b in bones])
    parents = np.array([indices[b.parent.name] if b.parent else -1 for b in bones])
    source_entry = next((e for e in json.loads(Path(io_path(OUT / 'sources.json')).read_text(encoding='utf-8')) if e['label'] == label))
    folder = Path(source_entry['folder'])
    info = source_entry['info']
    source_names = info['names']
    extra = {}
    with np.load(io_path(folder / info['asset_file'])) as source_asset, np.load(io_path(OUT / (label + '-' + kind + '-rotation.npz'))) as base:
        source_rest = source_asset['rest']
        source_ends = base_world0 = None
        source_frames = np.load(io_path(folder / info['frames_file']), mmap_mode='r')
        try:
            for (j, name) in enumerate(source_names[22:], 22):
                side = 'left' if name.startswith('Left') else 'right'
                hand = 20 if side == 'left' else 21
                suffix = name.split('Hand')[1]
                finger = suffix[:-1]
                phalanx = int(suffix[-1])
                target = None
                if kind in {'mixamo', 'warrok', 'ch14'}:
                    target = next((b.name for b in bones if b.name.split(':')[-1]==name), None)
                elif kind == 'vrm':
                    semantic = side + ('Little' if finger == 'Pinky' else finger) + ['Proximal', 'Intermediate', 'Distal'][phalanx - 1]
                    target = rig['semantic'].get(semantic)
                else:
                    jp = {'Thumb': '親指', 'Index': '人指', 'Middle': '中指', 'Ring': '薬指', 'Pinky': '小指'}[finger]
                    number = phalanx - 1 if finger == 'Thumb' else phalanx
                    target = jp + '０１２３'[number] + ('.L' if side == 'left' else '.R')
                if target not in indices:
                    continue
                target_bone = bones[indices[target]]
                vtarget = np.array(arm.matrix_world @ target_bone.tail_local) - rest[indices[target]]
                if phalanx < 3:
                    vsource = source_rest[j + 1] - source_rest[j]
                else:
                    if source_ends is None:source_ends = source_asset['ends']
                    vsource = source_ends[j]
                C = align(vtarget @ A, vsource)
                if base_world0 is None:base_world0 = base['world'][0].copy()
                Ch = source_frames[0, hand, :3, :3].T @ base_world0[hand]
                relative = source_frames[:, hand, :3, :3].swapaxes(-1, -2) @ source_frames[:, j, :3, :3]
                extra[target] = motion_world[:, hand] @ Ch.T @ relative @ C
        finally:source_frames._mmap.close()
    total_bones = len(bones)
    keyed = set(mapped) | set(extra) | ({'腰'} if kind == 'pmx' else set())
    if kind == 'pmx':
        validate_pmx_ancestor_controls(arm, keyed)
    needed = set(keyed)
    for name in keyed:
        b = arm.data.bones[name].parent
        while b is not None:
            needed.add(b.name)
            b = b.parent
    bones = [b for b in bones if b.name in needed]
    indices = {b.name: i for (i, b) in enumerate(bones)}
    rest = np.array([arm.matrix_world @ b.head_local for b in bones])
    R0 = np.array([rotation(np.array(arm.matrix_world @ b.matrix_local)[:3, :3]) for b in bones])
    parents = np.array([indices[b.parent.name] if b.parent else -1 for b in bones])
    p = np.zeros((n, len(bones), 3))
    rot = np.zeros((n, len(bones), 3, 3))
    channels = {}
    fk_start = time.perf_counter()
    original_length = np.array([b.length for b in bones])
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
        if pb.name in mapped or pb.name in extra or (kind == 'pmx' and pb.name.startswith('腰キャンセル')):
            for c in pb.constraints:
                c.mute = True
    for (j, b) in enumerate(bones):
        par = parents[j]
        parentR = np.tile(np.eye(3), (n, 1, 1)) if par < 0 else rot[:, par]
        before = R0[j] if par < 0 else R0[par].T @ R0[j]
        pre = parentR @ before
        if b.name in mapped:
            D = motion_world[:, mapped[b.name]]
            rot[:, j] = A @ D @ A.T @ R0[j]
        elif b.name in extra:
            rot[:, j] = A @ extra[b.name] @ A.T @ R0[j]
        elif kind == 'pmx' and b.name == '腰':
            rot[:, j] = A @ motion_world[:, 0] @ A.T @ R0[j]
        else:
            rot[:, j] = pre
        if par < 0:
            p[:, j] = rest[j]
        else:
            p[:, j] = p[:, par] + np.einsum('tij,j->ti', rot[:, par] @ R0[par].T, rest[j] - rest[par])
        if b.name in mapped or b.name in extra or (kind == 'pmx' and b.name == '腰'):
            basis = pre.swapaxes(-1, -2) @ rot[:, j]
            q = np.array([tuple(Matrix(m.tolist()).to_quaternion().normalized()) for m in basis])
            for t in range(1, n):
                if q[t] @ q[t - 1] < 0:
                    q[t] *= -1
            pb = arm.pose.bones[b.name]
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = q[0]
            pb.keyframe_insert('rotation_quaternion', frame=1)
            key_values(arm, pb.path_from_id('rotation_quaternion'), q)
    root_index = indices[motion_names[0]]
    shift = motion_root @ A.T - p[:, root_index]
    from .mapping import body_trajectory_root
    root_name = body_trajectory_root(
        {b.name: b.parent.name if b.parent else None for b in arm.data.bones}, mapped)
    if rig.get('trajectory_root', root_name) != root_name:
        raise ValueError('Character hierarchy changed after import. Import the character again.')
    trajectory_root = arm.data.bones[root_name]
    rest_to_world = np.array(arm.matrix_world @ trajectory_root.matrix_local)[:3, :3]
    root_location = shift @ np.linalg.inv(rest_to_world).T
    pb = arm.pose.bones[trajectory_root.name]
    assert not any((not c.mute and c.influence > 0 for c in pb.constraints)), 'Constrained trajectory root requires explicit handling'
    pb.location = root_location[0]
    pb.keyframe_insert('location', frame=1)
    key_values(arm, pb.path_from_id('location'), root_location)
    keyed.add(trajectory_root.name)
    assert all((c.data_path.startswith('pose.bones[') for c in curves(arm.animation_data.action)))
    timings['fk_and_action_seconds'] = time.perf_counter() - fk_start
    scene=bpy.context.scene
    with np.load(io_path(OUT/'motion-source.npz')) as source:fps=float(source['fps'])
    scene.frame_start=1;scene.frame_end=n;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
    scene.frame_set(1)
    save_blend(OUT/'animation.blend')
    return arm, sorted(keyed)

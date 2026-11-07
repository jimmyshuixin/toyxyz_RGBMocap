"""Verify replacing a managed face track preserves unrelated animation.

Run Blender -b --factory-startup --disable-autoexec --python-exit-code 1
--python this_file -- --output <new private directory>.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.face.blendshapes import BLENDSHAPE_NAMES
from mocap.retarget.blender_bake import curves
from mocap.retarget.blender_face import apply_face


def curve_snapshot(action):
    return {(c.data_path, c.array_index): [(tuple(p.co), tuple(p.handle_left),
            tuple(p.handle_right), p.interpolation) for p in c.keyframe_points]
            for c in curves(action)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    old, new = out/'old-generation', out/'new-generation'
    old.mkdir()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armdata = bpy.data.armatures.new('Rig')
    arm = bpy.data.objects.new('Rig', armdata)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bone = armdata.edit_bones.new('Root')
    bone.head, bone.tail = (0,0,0), (0,0,1)
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = bpy.data.meshes.new('Face')
    mesh.from_pydata([(0,0,1),(1,0,1),(0,0,2)], [], [(0,1,2)])
    obj = bpy.data.objects.new('Face', mesh)
    bpy.context.collection.objects.link(obj)
    for name in ('Basis', 'ワ', 'にやり', 'ArtistDetail'):
        obj.shape_key_add(name=name)
    keys = obj.data.shape_keys
    for frame, body, detail in ((1,0.,.2), (4,1.,.75)):
        arm.pose.bones['Root'].location.x = body
        arm.pose.bones['Root'].keyframe_insert(data_path='location', frame=frame)
        keys.key_blocks['ArtistDetail'].value = detail
        keys.key_blocks['ArtistDetail'].keyframe_insert(data_path='value', frame=frame)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end, scene.render.fps = 1, 4, 30
    (old/'import.json').write_text(json.dumps(dict(kind='mixamo')), encoding='utf8')
    (old/'bake.json').write_text(json.dumps(dict(frames=4,keyed_bones=['Root'])), encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(old/'animation.blend'))
    legacy = out/'legacy.npz'
    np.savez(legacy, morph_names=['ワ'], weights=np.full((4,1),.65,np.float32),
             fps=30.,start_seconds=0.,detected=np.ones(4,bool))
    apply_face(old,legacy,0.,4,30.)
    bpy.ops.wm.open_mainfile(filepath=str(old/'animation.blend'),load_ui=False)
    body_before = curve_snapshot(bpy.data.objects['Rig'].animation_data.action)
    keys = bpy.data.objects['Face'].data.shape_keys
    artist_path = keys.key_blocks['ArtistDetail'].path_from_id('value')
    artist_before = curve_snapshot(keys.animation_data.action)[artist_path,0]
    old_hashes = {path.name:digest(path) for path in old.iterdir() if path.is_file()}
    shutil.copytree(old,new)
    native = out/'native.npz'
    weights = np.zeros((4,52),np.float32)
    smiles = np.array([.1,.3,.5,.7],np.float32)
    for name in ('mouthSmileLeft','mouthSmileRight'):
        weights[:,BLENDSHAPE_NAMES.index(name)] = smiles
    np.savez(native,morph_names=BLENDSHAPE_NAMES,weights=weights,fps=30.,
             start_seconds=0.,detected=np.ones(4,bool))
    apply_face(new,native,0.,4,30.)
    bpy.ops.wm.open_mainfile(filepath=str(new/'animation.blend'),load_ui=False)
    arm = bpy.data.objects['Rig']
    keys = bpy.data.objects['Face'].data.shape_keys
    assert curve_snapshot(arm.animation_data.action) == body_before
    after = curve_snapshot(keys.animation_data.action)
    assert after[artist_path,0] == artist_before
    retired_path = keys.key_blocks['ワ'].path_from_id('value')
    assert (retired_path,0) not in after
    for frame in range(1,5):
        bpy.context.scene.frame_set(frame)
        assert keys.key_blocks['ワ'].value == 0
        np.testing.assert_allclose(keys.key_blocks['にやり'].value, smiles[frame-1],atol=1e-6)
    assert {path.name:digest(path) for path in old.iterdir() if path.is_file()} == old_hashes
    # Preflight errors must not publish changes even when a retired channel is
    # present. Artist drivers are never silently removed during replacement.
    for mode in ('driver','absolute'):
        folder = out/(mode+'-generation')
        shutil.copytree(old,folder)
        bpy.ops.wm.open_mainfile(filepath=str(folder/'animation.blend'),load_ui=False)
        keys = bpy.data.objects['Face'].data.shape_keys
        if mode == 'driver':
            keys.key_blocks['にやり'].driver_add('value')
        else:
            keys.use_relative = False
        bpy.ops.wm.save_as_mainfile(filepath=str(folder/'animation.blend'))
        before = digest(folder/'animation.blend')
        try:
            apply_face(folder,native,0.,4,30.)
        except ValueError as error:
            assert ('driver' if mode == 'driver' else 'Absolute') in str(error)
        else:
            raise AssertionError('Expected '+mode+' failure')
        assert digest(folder/'animation.blend') == before
    print('FACE_REPLACEMENT_PASS: retired managed channel cleared; new smile applied; '
          'artist/body curves and source generation preserved; driver/absolute failures retained',flush=True)


if __name__ == '__main__':
    main()

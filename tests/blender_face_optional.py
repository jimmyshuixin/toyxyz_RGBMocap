"""Blender check: optional face pass preserves body-only rigs; real errors fail.

Run Blender -b --factory-startup --disable-autoexec --python-exit-code 1
--python this_file -- --output <new private directory>.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mocap.retarget.blender_face import apply_face

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armdata=bpy.data.armatures.new('Rig');arm=bpy.data.objects.new('Rig',armdata)
    bpy.context.collection.objects.link(arm);bpy.context.view_layer.objects.active=arm;arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT');bone=armdata.edit_bones.new('Root');bone.head=(0,0,0);bone.tail=(0,0,1)
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh=bpy.data.meshes.new('Body');mesh.from_pydata([(0,0,1),(1,0,1),(0,0,2)],[],[(0,1,2)])
    obj=bpy.data.objects.new('Body',mesh);bpy.context.collection.objects.link(obj)
    for frame,x in [(1,0.),(4,1.)]:
        arm.pose.bones['Root'].location.x=x;arm.pose.bones['Root'].keyframe_insert(data_path='location',frame=frame)
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=4;scene.render.fps=30
    (out/'import.json').write_text(json.dumps({'kind':'mixamo'}),encoding='utf8')
    (out/'bake.json').write_text(json.dumps({'frames':4,'keyed_bones':['Root']}),encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'animation.blend'))
    before=hashlib.sha256((out/'animation.blend').read_bytes()).hexdigest()
    track=out/'face.npz';np.savez(track,morph_names=['あ'],weights=np.ones((4,1),np.float32),fps=30.,start_seconds=0.,detected=np.ones(4,bool))
    got,keyed,names=apply_face(out,track,0.,4,30.,optional=True)
    assert names==[] and keyed==['Root']
    assert hashlib.sha256((out/'animation.blend').read_bytes()).hexdigest()==before
    assert json.loads((out/'face-skipped.json').read_text(encoding='utf8'))['status']=='skipped'
    try:apply_face(out,track,0.,4,30.)
    except ValueError as exc:assert 'no supported facial shape keys' in str(exc)
    else:raise AssertionError('Explicit manual application must report missing shapes')
    # Adding supported shapes with drivers must not be silently skipped.
    obj=bpy.data.objects['Body'];obj.shape_key_add(name='Basis');shape=obj.shape_key_add(name='あ')
    shape.driver_add('value')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'animation.blend'))
    try:apply_face(out,track,0.,4,30.,optional=True)
    except ValueError as exc:assert 'driver' in str(exc)
    else:raise AssertionError('Optional mode must surface real facial driver errors')
    print('FACE_OPTIONAL_PASS: body-only animation unchanged; explicit missing-shape and driver failures retained',flush=True)

if __name__=='__main__':main()

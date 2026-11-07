"""Synthetic Blender face+body bake, sparse preview, VMD and FBX round trip.

Run with Blender --background --factory-startup --disable-autoexec
--python-exit-code 1 --python this_file -- --output <new private directory>.
No user character, model weight or video is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import addon_utils
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/character_import'))
from mocap.retarget.blender_face import apply_face
from mocap.retarget.blender_io import preview,main
from mocap.retarget.mapping import A

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable('mmd_tools',default_set=False,persistent=False)
armdata=bpy.data.armatures.new('Rig');arm=bpy.data.objects.new('Rig',armdata)
bpy.context.collection.objects.link(arm);bpy.context.view_layer.objects.active=arm;arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bone=armdata.edit_bones.new('Root');bone.head=(0,0,0);bone.tail=(0,0,1)
bpy.ops.object.mode_set(mode='OBJECT')
arm.pose.bones['Root'].mmd_bone.name_j='センター'
mesh=bpy.data.meshes.new('Face');mesh.from_pydata([(0,0,1),(1,0,1),(0,0,2)],[],[(0,1,2)]);mesh.update()
obj=bpy.data.objects.new('Face',mesh);bpy.context.collection.objects.link(obj)
group=obj.vertex_groups.new(name='Root');group.add([0,1,2],1.,'REPLACE')
modifier=obj.modifiers.new('Armature','ARMATURE');modifier.object=arm
obj.shape_key_add(name='Basis')
names=['まばたき','ウィンク','ウィンク右','あ','ワ']
for i,name in enumerate(names):
    key=obj.shape_key_add(name=name);key.data[i%3].co.y+=.01*(i+1)
pb=arm.pose.bones['Root']
for frame,x in [(1,0.),(10,1.)]:
    pb.location.x=x;pb.keyframe_insert(data_path='location',frame=frame)
for curve in arm.animation_data.action.fcurves:
    for point in curve.keyframe_points:point.interpolation='LINEAR'
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=10;scene.render.fps=30;scene.frame_set(1)
(out/'import.json').write_text(json.dumps(dict(kind='pmx',mmd_import_scale=.08)),encoding='utf-8')
(out/'pmx-rig.json').write_text(json.dumps(dict(trajectory_root='Root',armature='Rig')),encoding='utf-8')
(out/'bake.json').write_text(json.dumps(dict(keyed_bones=['Root'],frames=10,object_animation_channels=0)),encoding='utf-8')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'animation.blend'))
weights=np.zeros((10,5),np.float32);weights[:,0]=np.linspace(0,1,10);weights[:,3]=np.linspace(1,0,10)
track=out/'capture.npz';np.savez(track,morph_names=names,weights=weights,fps=30.,start_seconds=4.,detected=np.ones(10,bool))
arm,keyed,channels=apply_face(out,track,4.,10,30.)
body=[]
for frame in range(1,11):
    scene=bpy.context.scene;scene.frame_set(frame);body.append(arm.pose.bones['Root'].location.x)
np.testing.assert_allclose(body,np.linspace(0,1,10),atol=1e-6)
preview(out,arm,keyed,10)
with np.load(out/'preview.npz') as data:
    asset={k:data[k] for k in data.files}
frames=np.load(out/'preview-frames.npy')
assert asset['morph_entries'].shape==(5,4)
for i in (0,4,9):
    scene.frame_set(i+1);bpy.context.view_layer.update()
    evaluated=bpy.data.objects['Face'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    actual=np.array([evaluated.matrix_world@v.co for v in evaluated.data.vertices])@A
    result=[]
    for v in range(3):
        p=asset['vertices'][v].copy();first,count=asset['morph_rows'][v]
        for ch,dx,dy,dz in asset['morph_entries'][first:first+count]:
            p+=frames[i].ravel()[int(asset['morph_weight_offset'])+int(ch)]*np.array([dx,dy,dz])
        result.append((frames[i,0]@np.r_[p,1])[:3])
    np.testing.assert_allclose(result,actual,atol=1e-6)
main(['export',str(out),str(out),str(out/'merged.vmd'),'null','Synthetic face',json.dumps(dict(format='vmd'))])
from mmd_tools.core import vmd
parsed=vmd.File();parsed.load(filepath=str(out/'merged.vmd'))
assert len(parsed.shapeKeyAnimation)==5
for name,column in [('まばたき',0),('あ',3)]:
    np.testing.assert_allclose([key.weight for key in parsed.shapeKeyAnimation[name]],weights[:,column])
main(['export',str(out),str(out),str(out/'merged.fbx'),'null','Synthetic face',json.dumps(dict(format='fbx',include_mesh=True))])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(out/'merged.fbx'))
face=next(o for o in bpy.context.scene.objects if o.type=='MESH')
for frame in (1,5,10):
    bpy.context.scene.frame_set(frame)
    np.testing.assert_allclose(face.data.shape_keys.key_blocks['あ'].value,weights[frame-1,3],atol=1e-5)
print('FACE_INTEGRATION_PASS: body preserved; sparse preview matches Blender; VMD and mesh FBX round trips passed',flush=True)

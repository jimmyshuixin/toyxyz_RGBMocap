"""Native MediaPipe track -> animated rig, sparse preview, FBX and MMD VMD.

Run Blender --background --factory-startup --disable-autoexec
--python-exit-code 1 --python this_file -- --track <face.npz>
--output <new private directory>. The supplied track must have 52 native channels.
Creates synthetic rigs only; never changes the supplied track or user characters.
"""
import argparse
import json
from pathlib import Path
import sys

import addon_utils
import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/character_import'))
from mocap.retarget.blender_face import apply_face
from mocap.retarget.blender_io import main as blender_io, preview
from mocap.retarget.face_track import load_track, map_shape_channels
from mocap.retarget.mapping import A
from mocap.face.vmd import export_vmd


def create_rig(folder, names, frames, fps, *, pmx=False):
    folder.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if pmx:addon_utils.enable('mmd_tools',default_set=False,persistent=False)
    armdata=bpy.data.armatures.new('Rig')
    arm=bpy.data.objects.new('Rig',armdata)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active=arm;arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bone=armdata.edit_bones.new('Root');bone.head=(0,0,0);bone.tail=(0,0,1)
    bpy.ops.object.mode_set(mode='OBJECT')
    if pmx:arm.pose.bones['Root'].mmd_bone.name_j='センター'
    mesh=bpy.data.meshes.new('Face')
    mesh.from_pydata([(0,0,1),(1,0,1),(0,0,2)],[],[(0,1,2)]);mesh.update()
    obj=bpy.data.objects.new('Face',mesh);bpy.context.collection.objects.link(obj)
    group=obj.vertex_groups.new(name='Root');group.add([0,1,2],1.,'REPLACE')
    obj.modifiers.new('Armature','ARMATURE').object=arm
    obj.shape_key_add(name='Basis')
    for i,name in enumerate(names):
        shape=obj.shape_key_add(name=name);shape.data[i%3].co.y+=.002*(i+1)
    for frame,x in ((1,0.),(frames,1.)):
        arm.pose.bones['Root'].location.x=x
        arm.pose.bones['Root'].keyframe_insert(data_path='location',frame=frame)
    for curve in arm.animation_data.action.fcurves:
        for point in curve.keyframe_points:point.interpolation='LINEAR'
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=frames
    scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps;scene.frame_set(1)
    (folder/'import.json').write_text(json.dumps(dict(kind='pmx' if pmx else 'mixamo',mmd_import_scale=.08)),encoding='utf8')
    (folder/'bake.json').write_text(json.dumps(dict(keyed_bones=['Root'],frames=frames,object_animation_channels=0)),encoding='utf8')
    if pmx:(folder/'pmx-rig.json').write_text(json.dumps(dict(trajectory_root='Root',armature='Rig')),encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'animation.blend'))
    return arm


def verify_preview(folder, arm, keyed, count):
    body=[]
    for frame in range(1,count+1):
        bpy.context.scene.frame_set(frame);body.append(arm.pose.bones['Root'].location.x)
    np.testing.assert_allclose(body,np.linspace(0,1,count),atol=1e-6)
    preview(folder,arm,keyed,count)
    with np.load(folder/'preview.npz') as data:asset={k:data[k] for k in data.files}
    frames=np.load(folder/'preview-frames.npy')
    maximum=0.
    for frame in sorted({0,count//2,count-1}):
        bpy.context.scene.frame_set(frame+1);bpy.context.view_layer.update()
        evaluated=bpy.data.objects['Face'].evaluated_get(bpy.context.evaluated_depsgraph_get())
        actual=np.asarray([evaluated.matrix_world@v.co for v in evaluated.data.vertices])@A
        expected=[]
        for vertex in range(3):
            p=asset['vertices'][vertex].copy();first,n=asset['morph_rows'][vertex]
            for channel,dx,dy,dz in asset['morph_entries'][first:first+n]:
                p+=frames[frame].ravel()[int(asset['morph_weight_offset'])+int(channel)]*np.asarray([dx,dy,dz])
            expected.append((frames[frame,0]@np.r_[p,1])[:3])
        np.testing.assert_allclose(expected,actual,atol=1e-6)
        maximum=max(maximum,float(np.max(np.abs(np.asarray(expected)-actual))))
    return maximum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--track',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    track=load_track(args.track);names=track['morph_names'];weights=track['weights']
    assert len(names)==52 and names[0]=='_neutral'
    n=len(weights);fps=track['fps'];assert n>1
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    native=out/'native';create_rig(native,names,n,fps)
    arm,keyed,channels=apply_face(native,args.track,track['start_seconds'],n,fps)
    assert len(channels)==51 and '_neutral' not in channels
    np.testing.assert_equal(bpy.data.objects['Face'].data.shape_keys.key_blocks['_neutral'].value,0)
    error=verify_preview(native,arm,keyed,n)
    blender_io(['export',str(native),str(native),str(native/'merged.fbx'),'null','Native expression rig',json.dumps(dict(format='fbx',include_mesh=True))])
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(native/'merged.fbx'))
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    for frame in sorted({0,n//2,n-1}):
        bpy.context.scene.frame_set(frame+1)
        for column,name in enumerate(names):
            if name=='_neutral':continue
            np.testing.assert_allclose(obj.data.shape_keys.key_blocks[name].value,weights[frame,column],atol=1e-5)
    # PMX named morph mapping and VMD export must retain the same body animation.
    mmd_names=['まばたき','ウィンク','ウィンク右','あ','い','う','お','にやり','への字','上','下','困る']
    pmx=out/'pmx';create_rig(pmx,mmd_names,n,fps,pmx=True)
    arm,keyed,channels=apply_face(pmx,args.track,track['start_seconds'],n,fps)
    assert len(channels)>5
    pmx_error=verify_preview(pmx,arm,keyed,n)
    blender_io(['export',str(pmx),str(pmx),str(pmx/'merged.vmd'),'null','Native mapped PMX',json.dumps(dict(format='vmd'))])
    from mmd_tools.core import vmd
    parsed=vmd.File();parsed.load(filepath=str(pmx/'merged.vmd'))
    expected=map_shape_channels(names,weights,set(mmd_names))
    left=weights[:,names.index('eyeBlinkLeft')];right=weights[:,names.index('eyeBlinkRight')]
    np.testing.assert_allclose(expected['まばたき']+expected['ウィンク'],left,atol=1e-7)
    np.testing.assert_allclose(expected['まばたき']+expected['ウィンク右'],right,atol=1e-7)
    assert set(parsed.shapeKeyAnimation)==set(expected)
    for name,values in expected.items():
        np.testing.assert_allclose([key.weight for key in parsed.shapeKeyAnimation[name]],values,atol=1e-6)
    standalone=export_vmd(args.track,out/'facial-only.vmd')
    parsed_face=vmd.File();parsed_face.load(filepath=str(standalone))
    assert len(parsed_face.shapeKeyAnimation)>5 and not parsed_face.boneAnimation
    summary=dict(frames=n,native_channels=51,pmx_channels=channels,
        preview_max_error=max(error,pmx_error),fbx_roundtrip=True,
        standalone_vmd_channels=list(parsed_face.shapeKeyAnimation),body_unchanged=True)
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('NATIVE_FACE_INTEGRATION_PASS '+json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':main()

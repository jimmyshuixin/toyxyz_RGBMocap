"""Actor identity, optional stage receipts and automatic retarget selection."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mocap.face import pipeline
from mocap.storage import atomic_json


class FacePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name)
        self.state=dict(frames=4,fps=10.,start=15.,end=15.4,width=640,height=480,
                        person_id=3,pipeline={'model':'exmr'},face_options={'enabled':True,'smoothing':True})
        atomic_json(self.project/'project.json',self.state)
        (self.project/'input.mp4').write_bytes(b'unchanged prepared CFR clip')
        self.boxes=np.array([[100,20,300,420],[300,10,450,450],[0,0,50,40],[1,2,3,4]],float)
        np.savez(self.project/'tracking.npz',boxes=self.boxes,observed=np.array([True,False,True,True]))
        self.events=[]
        self.emit=lambda kind,**kw:self.events.append(dict(type=kind,**kw))
        self.calls=[]

    def fake_capture(self,job,progress,**kwargs):
        self.calls.append((job,kwargs))
        out=Path(job['output_dir']);out.mkdir(parents=True)
        np.savez(out/'face.npz',morph_names=['あ'],weights=np.full((4,1),.5,np.float32),
                 fps=10.,start_seconds=15.,detected=np.ones(4,bool))
        return dict(npz=str(out/'face.npz'),detected_frames=4,frame_count=4)

    def capture(self,**kwargs):
        with patch('mocap.face.runtime.model_path',return_value=self.project/'model.task'), \
             patch('mocap.face.capture.run',side_effect=self.fake_capture):
            return pipeline.run_project(self.project,emit=self.emit,**kwargs)

    def test_dynamic_crops_bind_actor_and_skip_unobserved_frames(self):
        crop=pipeline.actor_crops(self.project,self.state)
        np.testing.assert_allclose(crop(0,(480,640,3)),[100/640,20/480,300/640,240/480])
        self.assertIsNone(crop(1,(480,640,3)))
        self.assertIsNone(crop(3,(480,640,3)))
        self.assertIsNone(crop(4,(480,640,3)))

    def test_atomic_track_reuse_and_body_fields_do_not_invalidate(self):
        before=(self.project/'project.json').read_bytes()
        receipt=self.capture()
        track=pipeline.load_current(self.project)
        self.assertEqual(track,self.project/receipt['track'])
        self.assertEqual(before,(self.project/'project.json').read_bytes())
        self.assertTrue(self.calls[0][0]['reject_multiple_faces'])
        self.assertEqual(self.calls[0][0]['timeline_offset_seconds'],15.)
        self.assertEqual(self.calls[0][0]['start_seconds'],0.)
        self.assertIsNone(self.calls[0][1]['crop_for_frame'](1,(480,640,3)))
        self.capture();self.assertEqual(len(self.calls),1)
        self.state['correction']={'enabled':True};atomic_json(self.project/'project.json',self.state)
        self.assertEqual(pipeline.load_current(self.project),track)

    def test_disabled_preserves_history_and_stops_auto_selection(self):
        self.capture();track=pipeline.load_current(self.project)
        pipeline.run_project(self.project,{'enabled':False},self.emit)
        self.assertIsNone(pipeline.load_current(self.project));self.assertTrue(track.is_file())
        self.assertEqual(json.loads((self.project/'face_capture.json').read_text())['status'],'disabled')

    def test_no_detection_deactivates_previous_generation_with_warning(self):
        self.capture();track=pipeline.load_current(self.project)
        with patch('mocap.face.runtime.model_path',return_value=self.project/'model.task'), \
             patch('mocap.face.capture.run',side_effect=RuntimeError('No face was detected.')):
            result=pipeline.run_project(self.project,emit=self.emit,force=True)
        self.assertEqual(result['status'],'skipped');self.assertTrue(track.is_file())
        self.assertIsNone(pipeline.load_current(self.project))
        self.assertIn('Body motion is ready',self.events[-1]['message'])

    def test_identity_tracking_source_and_settings_changes_invalidate(self):
        self.capture()
        for key,value in [('person_id',4),('start',16.),('face_options',{'enabled':False})]:
            old=self.state[key];self.state[key]=value;atomic_json(self.project/'project.json',self.state)
            self.assertIsNone(pipeline.load_current(self.project))
            self.state[key]=old;atomic_json(self.project/'project.json',self.state)
        self.assertIsNotNone(pipeline.load_current(self.project))
        (self.project/'input.mp4').write_bytes(b'different body input')
        self.assertIsNone(pipeline.load_current(self.project))

    def test_unsupported_model_is_skipped_without_guessing_face_identity(self):
        self.state['pipeline']['model']='prompthmr';atomic_json(self.project/'project.json',self.state)
        result=pipeline.run_project(self.project,emit=self.emit)
        self.assertEqual(result['status'],'skipped')
        self.assertIsNone(pipeline.load_current(self.project))

    def test_escape_receipt_rejected(self):
        receipt=self.capture();receipt['track']='../outside/face.npz'
        atomic_json(self.project/'face_capture.json',receipt)
        self.assertIsNone(pipeline.load_current(self.project))

    def test_mid_capture_source_change_never_activates_new_track(self):
        def changed(*args,**kwargs):
            result=self.fake_capture(*args,**kwargs)
            (self.project/'input.mp4').write_bytes(b'changed during capture')
            return result
        with patch('mocap.face.runtime.model_path',return_value=self.project/'model.task'), \
             patch('mocap.face.capture.run',side_effect=changed):
            with self.assertRaisesRegex(RuntimeError,'changed during'):
                pipeline.run_project(self.project,emit=self.emit)
        self.assertFalse((self.project/'face_capture.json').exists())

    def test_retarget_consumes_each_source_face_and_marks_it_optional(self):
        from mocap.retarget import worker
        from mocap.retarget.source import stamp
        asset=self.project/'asset';asset.mkdir()
        (asset/'import.json').write_text('{"kind":"pmx"}',encoding='utf8')
        for name in ('pmx-rest.blend','pmx-rig.json','pmx-canonical.npz','pmx-contact-geometry.npz'):
            (asset/name).write_bytes(b'fixture')
        from mocap.retarget.mapping import PMX, PARENTS
        atomic_json(asset/'pmx-rig.json',dict(bones=[dict(name=name) for name in PMX]))
        np.savez(asset/'pmx-canonical.npz',names=PMX,parents=PARENTS)
        sources=[self.project/'actor-a',self.project/'actor-b']
        faces=[]
        for index,source in enumerate(sources):
            source.mkdir();(source/'project.json').write_text('{}',encoding='utf8')
            face=source/'face.npz';face.write_bytes(bytes([index+1]));faces.append(face)
        job=dict(mode='retarget',output=str(self.project/'retargeted'),asset=str(asset),result='initial',source_name='Actor',
                 takes=[dict(project=str(source),start=15.) for source in sources])
        request=self.project/'request.json';atomic_json(request,job)
        invoked=[]
        def prepare(source,kind,out):
            return dict(frames=4,fps=10.,source_stats=stamp([source/'project.json']))
        def invoke(mode,out,*args):
            if mode=='face':invoked.append((Path(args[0]),json.loads(args[1])))
        with patch('mocap.retarget.source.prepare',side_effect=prepare), \
             patch('mocap.retarget.pipeline.run',return_value={}), \
             patch.object(worker,'invoke',side_effect=invoke), \
             patch('mocap.face.pipeline.load_current',side_effect=lambda p:Path(p)/'face.npz'):
            worker.main(['--request',str(request)])
        self.assertEqual([entry[0] for entry in invoked],faces)
        self.assertTrue(all(entry[1]['optional'] for entry in invoked))
        for index,face in enumerate(faces):
            receipt=json.loads((self.project/f'retargeted/take-{index}/receipt.json').read_text(encoding='utf8'))
            self.assertEqual(receipt['face_capture']['source'],str(face))
            self.assertIn(str(face),receipt['source_stats'])


if __name__=='__main__':unittest.main()

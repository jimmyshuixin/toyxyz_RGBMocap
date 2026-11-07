"""Saved camera compatibility across the captured-world heading change."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    import PySide6
except ImportError:
    from mocap.runtime_registry import activate
    activate(ROOT, 'gui')

from mocap.motion_heading import HEADING_VERSION, upgrade_saved_camera
from mocap.project_workspace_ui import ModelWorkspace
from mocap.project_save_ui import ProjectSaving


def legacy_camera():
    return dict(yaw=-2.4, pitch=.2, distance=3.7,
                pan=[.25, .1, -.3], scene_center=[5., .85, -5.],
                target=[5.25, .95, -5.3], extra={'retained': [2, 4]})


class View:
    def __init__(self):
        self.yaw=.3; self.pitch=.2; self.distance=4.
        self.pan=np.zeros(3); self.scene_center=np.array([-5., .85, 5.])
        self.root=np.array([-5., 1., 5.]); self.auto_fit=True

    def target(self):
        result=self.scene_center.copy()
        result[[0, 2]]=self.root[[0, 2]]
        return result+self.pan

    def update(self):
        pass


def camera_space(camera, points):
    """Independent look-at construction for the saved orbit camera."""
    yaw, pitch=camera['yaw'], camera['pitch']
    target=np.asarray(camera['target'])
    offset=camera['distance']*np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    eye=target+offset
    forward=target-eye; forward/=np.linalg.norm(forward)
    right=np.cross(forward, [0., 1., 0.]); right/=np.linalg.norm(right)
    up=np.cross(right, forward)
    return (points-eye)@np.array([right, up, forward]).T


class SavedHeadingCameraTests(unittest.TestCase):
    def test_old_camera_preserves_projected_composition_and_input(self):
        camera=legacy_camera(); before=copy.deepcopy(camera)
        updated=upgrade_saved_camera(camera)
        points=np.array([[5., 1., -5.], [5.4, 1.7, -4.8], [4.6, .1, -5.3]])
        turned=points.copy(); turned[:, [0, 2]]*=-1
        np.testing.assert_allclose(camera_space(updated, turned),camera_space(camera, points),atol=2e-14)
        self.assertEqual(camera, before)
        self.assertEqual(updated['heading_version'],HEADING_VERSION)
        self.assertEqual(updated['extra'],camera['extra'])
        updated['extra']['retained'].append(8)
        self.assertEqual(camera, before)

    def test_new_camera_roundtrip_is_idempotent_and_detached(self):
        camera=upgrade_saved_camera(legacy_camera())
        restored=upgrade_saved_camera(json.loads(json.dumps(camera)))
        self.assertEqual(restored,camera)
        restored['pan'][0]=123
        self.assertNotEqual(restored['pan'],camera['pan'])
        # Save-as keeps NumPy arrays in memory rather than a JSON-only shape.
        camera['pan']=np.array(camera['pan'])
        restored=upgrade_saved_camera(camera); restored['pan'][0]=456
        self.assertNotEqual(restored['pan'][0],camera['pan'][0])

    def test_legacy_camera_without_target_is_supported(self):
        camera=legacy_camera(); del camera['target']
        result=upgrade_saved_camera(camera)
        self.assertNotIn('target',result)
        self.assertEqual(result['scene_center'],[-5.,.85,5.])
        self.assertEqual(result['pan'],[-.25,.1,.3])
        self.assertAlmostEqual(result['yaw'],camera['yaw']+np.pi)

    def test_workspace_restore_keeps_distant_actor_centered_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'project.json'
            path.write_text(json.dumps({'view':{'camera':legacy_camera()}}),encoding='utf-8')
            original=path.read_bytes()
            camera=json.loads(original)['view']['camera']; before=copy.deepcopy(camera)
            host=SimpleNamespace(motion_view=View(),_workspace_camera=camera,people_root=None,scene_data=None)
            for _ in range(3):
                ModelWorkspace.restore_workspace_camera(host)
                np.testing.assert_allclose(host.motion_view.target(),[-5.25,.95,5.3],atol=1e-14)
                np.testing.assert_allclose(host.motion_view.pan,[-.25,.1,.3],atol=1e-14)
                self.assertAlmostEqual(host.motion_view.yaw,-2.4+np.pi)
            self.assertFalse(host.motion_view.auto_fit)
            self.assertFalse(hasattr(host.motion_view,'heading_version'))
            self.assertEqual(camera,before)
            self.assertEqual(path.read_bytes(),original)

    def test_workspace_save_marks_new_camera_and_reloads_once(self):
        view=View(); view.pan=np.array([.7,.1,-.2])
        host=SimpleNamespace(motion_view=view,info={'fps':30.},source_offset=1.,
            slider=SimpleNamespace(value=lambda:5,clip_in=0,clip_out=30),
            camera_tools=SimpleNamespace(background_options=lambda:{}))
        saved=ModelWorkspace.workspace_view(host)
        self.assertEqual(saved['camera']['heading_version'],HEADING_VERSION)
        fresh=SimpleNamespace(motion_view=View(),_workspace_camera=saved['camera'],people_root=None,scene_data=None)
        ModelWorkspace.restore_workspace_camera(fresh,consume=True)
        self.assertIsNone(fresh._workspace_camera)
        self.assertEqual(fresh.motion_view.yaw,view.yaw)
        np.testing.assert_allclose(fresh.motion_view.target(),view.target())

    def test_save_as_temporary_view_is_marked_before_reload(self):
        view=View(); submitted=[]
        host=SimpleNamespace(busy=False,contact_saving=False,mamma_pending=False,video_path=Path('input.mp4'),
            contact_tracks={},workspace_root=None,project=None,people_root=None,motion_view=view,
            stop_playback=lambda:None,selected_result=lambda:'initial',
            model_choice=SimpleNamespace(currentData=lambda:'exmr'),info={'fps':30.},
            workspace_view=lambda:{'camera':{'heading_version':HEADING_VERSION}},workspace_settings=lambda:{},
            selected_face_options=lambda:{},mamma_inputs=[],mamma_active_camera=0,selected_flow_options=lambda:{},
            exmr_camera=SimpleNamespace(currentData=lambda:True),exmr_smoothing=SimpleNamespace(isChecked=lambda:True),
            selected_correction_options=lambda:{},slider=SimpleNamespace(value=lambda:0,clip_in=0,clip_out=10),
            start_jobs=submitted.append)
        try:
            with patch('mocap.project_save.validate_destination',return_value=Path('output')):
                self.assertTrue(ProjectSaving.submit_project_save(host,Path('output')))
            self.assertEqual(host._project_save_view['heading_version'],HEADING_VERSION)
            self.assertEqual(upgrade_saved_camera(host._project_save_view)['yaw'],view.yaw)
            request=json.loads(host._project_save_request.read_text(encoding='utf-8'))
            self.assertEqual(request['workspace_view']['camera']['heading_version'],HEADING_VERSION)
            self.assertEqual(len(submitted),1)
        finally:
            ProjectSaving.cleanup_project_save(host)

    def test_save_as_finish_does_not_rotate_new_temporary_camera(self):
        view=View(); snapshot={key:copy.deepcopy(getattr(view,key)) for key in ('yaw','pitch','distance','pan','scene_center')}
        snapshot['heading_version']=HEADING_VERSION
        host=SimpleNamespace(_project_saved_root=Path('copy'),_project_save_view=snapshot,motion_view=View(),
            load_project=lambda root:None,status=SimpleNamespace(setText=lambda text:None),
            task_info=SimpleNamespace(text=lambda:'',setText=lambda text:None),
            progress=SimpleNamespace(setRange=lambda *args:None,setValue=lambda value:None),
            cleanup_project_save=lambda:None,update_readiness=lambda:None)
        ProjectSaving.finish_project_save(host)
        self.assertEqual(host.motion_view.yaw,view.yaw)
        np.testing.assert_array_equal(host.motion_view.scene_center,view.scene_center)
        self.assertFalse(hasattr(host.motion_view,'heading_version'))


if __name__=='__main__':
    unittest.main()

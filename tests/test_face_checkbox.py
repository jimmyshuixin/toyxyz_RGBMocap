"""Actual Qt checkbox, reconstruction queue and project settings regressions.

Body model execution is mocked; no user project, video, model or runtime is
changed. Queue construction and JSON persistence use production methods.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    import PySide6
except ImportError:
    from mocap.runtime_registry import activate
    activate(ROOT, 'gui')
from PySide6.QtWidgets import QApplication
from mocap.gui import MainWindow
from mocap.project_workspace import create, read


class FaceCheckboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        window = self.window
        window.workspace_root = None
        window.project = None
        window.people_root = None
        window._people_batch = False
        window.busy = False
        window.close()
        self.app.processEvents()

    def select_without_starting_models(self, name):
        window = self.window
        blocked = window.model_choice.blockSignals(True)
        window.model_choice.setCurrentIndex(window.model_choice.findData(name))
        window.model_choice.blockSignals(blocked)
        window.update_readiness()

    def test_default_and_supported_model_controls(self):
        window = self.window
        self.assertFalse(window.face_capture.isChecked())
        for model in ('exmr', 'flowhmr', 'gvhmr'):
            self.select_without_starting_models(model)
            self.assertTrue(window.face_capture.isEnabled(), model)
        window.face_capture.setChecked(True)
        for model in ('mamma', 'prompthmr', 'pear'):
            self.select_without_starting_models(model)
            self.assertFalse(window.face_capture.isEnabled(), model)
            self.assertFalse(window.selected_face_options()['enabled'], model)
        self.assertIsNone(window.process)

    def test_reconstruct_snapshots_checkbox_options(self):
        window = self.window
        window.face_capture.setChecked(True)
        with patch('mocap.face.runtime.status', return_value={'ready': True, 'message': 'Ready'}), \
             patch.object(window, 'input_jobs') as submit:
            window.run_full()
        self.assertEqual(submit.call_args.kwargs['face_options'], {'enabled': True, 'smoothing': True})
        window.face_capture.setChecked(False)
        with patch('mocap.face.runtime.status') as status, patch.object(window, 'input_jobs') as submit:
            window.run_full()
        status.assert_not_called()
        self.assertFalse(submit.call_args.kwargs['face_options']['enabled'])

    def test_missing_optional_runtime_stops_before_body_work(self):
        window = self.window
        window.face_capture.setChecked(True)
        with patch('mocap.face.runtime.status', return_value={'ready': False, 'message': 'Missing'}), \
             patch.object(window, 'input_jobs') as submit:
            window.run_full()
        submit.assert_not_called()
        self.assertIn('Set Up Face Capture', window.status.text())

    def test_only_reconstruct_request_adds_face_to_single_actor_queue(self):
        window = self.window
        window.video_path = Path('clip.mp4')
        window.info = {'fps': 30., 'duration': 2., 'frames': 60}
        window.video_readable = True
        window.face_capture.setChecked(True)
        stages = ['track', 'exmr_preprocess', 'exmr_features', 'exmr', 'export']
        for enabled in (True, False):
            with patch('mocap.gui.missing', return_value=[]), \
                 patch.object(window, 'processing_range', return_value=(0., 2.)), \
                 patch.object(window, 'input_person_jobs', return_value=False), \
                 patch.object(window, 'start_jobs') as submit:
                window.input_jobs(stages, result='initial', face_options={'enabled': enabled, 'smoothing': True})
            jobs = submit.call_args.args[0]
            self.assertEqual(jobs[0][0], 'prepare')
            self.assertEqual(sum(stage == 'face_capture' for stage, _ in jobs), int(enabled))
            self.assertEqual(jobs[-1][0], 'face_capture' if enabled else 'export')
        with patch.object(window, 'input_jobs') as submit:
            window.run_analysis()
        self.assertNotIn('face_options', submit.call_args.kwargs)
        with patch.object(window, 'next_job'):
            window.start_jobs([('export', ['--project', 'prepared'])])
        self.assertEqual(window.jobs, [('export', ['--project', 'prepared'])])
        self.assertIsNone(window._batch_face_options)

    def test_workspace_and_manifest_settings_round_trip(self):
        window = self.window
        window.restore_workspace_settings({'face_options': {'enabled': True, 'smoothing': True}})
        self.assertTrue(window.face_capture.isChecked())
        self.assertTrue(window.workspace_settings()['face_options']['enabled'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create(root, root/'clip.mp4', {'path': str(root/'clip.mp4')}, 'exmr')
            project = root/'exmr'/'take'
            project.mkdir(parents=True)
            original = {'pipeline': {'model': 'exmr', 'result': 'initial'}, 'stages': {'exmr': {'fingerprint': 'unchanged'}}}
            (project/'project.json').write_text(json.dumps(original), encoding='utf8')
            window.workspace_root, window.project = root, project
            window.remember_workspace_model()
            state = read(root)
            saved = json.loads((project/'project.json').read_text(encoding='utf8'))
            self.assertTrue(state['models']['exmr']['settings']['face_options']['enabled'])
            self.assertTrue(saved['face_options']['enabled'])
            self.assertEqual(saved['stages'], original['stages'])
            window.restore_face_options({})
            self.assertFalse(window.face_capture.isChecked())
            window.restore_workspace_settings(state['models']['exmr']['settings'])
            self.assertTrue(window.face_capture.isChecked())
            window.workspace_root = window.project = None

    def test_prepare_persistence_uses_batch_snapshot_not_changed_checkbox(self):
        window = self.window
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project/'project.json').write_text('{"pipeline":{"model":"exmr"}}', encoding='utf8')
            window._batch_face_options = {'enabled': True, 'smoothing': True}
            window.face_capture.setChecked(False)
            window.save_batch_face_options(project)
            self.assertTrue(json.loads((project/'project.json').read_text(encoding='utf8'))['face_options']['enabled'])

    def test_people_queue_targets_selected_takes_after_body_work(self):
        window = self.window
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            projects = [root/'person_1'/'take_0', root/'person_2'/'take_0']
            for project in projects:
                project.mkdir(parents=True)
                (project/'project.json').write_text('{"pipeline":{"model":"exmr"}}', encoding='utf8')
            window.people_root = root
            window._people_stages = ['exmr_preprocess', 'exmr_features', 'exmr', 'correct', 'export', 'face_capture']
            window._people_result = 'corrected_initial'
            window._people_face_options = {'enabled': True, 'smoothing': True}
            window._batch_face_options = dict(window._people_face_options)
            window._people_inference_jobs = []; window._people_post_jobs = []
            window.jobs = [('person', ['--person-id', '2'])]
            window.people_event({'type': 'person_prepared', 'projects': [str(projects[0])]})
            self.assertEqual(window.jobs[0][0], 'person')
            window.jobs = []
            window.people_event({'type': 'person_prepared', 'projects': [str(projects[1])]})
            self.assertEqual(window.jobs[0][0], 'exmr_batch')
            faces = [(stage, args) for stage, args in window.jobs if stage == 'face_capture']
            self.assertEqual([args[1] for _, args in faces], list(map(str, projects)))
            for _, args in faces:
                self.assertNotIn('--result', args)
                self.assertTrue(json.loads(args[args.index('--face-options')+1])['enabled'])
                i = window.jobs.index(('face_capture', args))
                self.assertEqual(window.jobs[i-1][0], 'export')
            window.people_root = None

    def test_face_only_completion_without_export_json_is_safe(self):
        window = self.window
        with tempfile.TemporaryDirectory() as temporary:
            window.project = Path(temporary)
            window.jobs = []; window.busy = True; window.current_stage = 'face_capture'
            with patch.object(window, 'finish_single_tracking'), \
                 patch.object(window, 'finish_people_jobs'), \
                 patch.object(window, 'refresh_results'), \
                 patch.object(window, 'update_readiness'), \
                 patch.object(window, 'next_character_refresh'), \
                 patch.object(window, 'set_active_result') as select:
                window.next_job()
            select.assert_not_called()
            self.assertFalse(window.busy)
            window.project = None


if __name__ == '__main__':
    unittest.main()

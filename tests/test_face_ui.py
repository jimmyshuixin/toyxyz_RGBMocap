"""GUI orchestration/timing and independent binary facial VMD checks."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
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
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtTest import QTest
from mocap.face_ui import FaceCaptureDialog, capture_source
from mocap.face.vmd import export_vmd


class FaceUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, folder):
        parent=QWidget();base=Path(folder)
        parent.video_path=None;parent.info=None;parent.project=None;parent.workspace_root=None
        parent.characters=[];parent.selected_character=lambda:None
        parent.character_path=lambda relative:base/relative
        return parent,FaceCaptureDialog(parent)

    def wait_process(self, dialog):
        deadline=time.monotonic()+10
        while dialog.process is not None and time.monotonic()<deadline:QTest.qWait(20)
        self.assertIsNone(dialog.process,'Worker did not finish or was not reaped')

    def test_process_completion_is_not_shadowed_by_qdialog_finished_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,dialog=self.dialog(folder)
            dialog.start_process([sys.executable,'-X','utf8','-c','print("setup complete")'],'setup')
            self.wait_process(dialog)
            self.assertIn('setup finished',dialog.status.text())
            self.assertTrue(dialog.setup.isEnabled())
            dialog.close();parent.close()

    def test_immediate_cancel_reaps_starting_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,dialog=self.dialog(folder)
            dialog.start_process([sys.executable,'-X','utf8','-c','import time;time.sleep(30)'],'capture')
            dialog.reject()
            self.wait_process(dialog)
            self.assertTrue(dialog.cancelled)
            self.assertIn('Cancelled',dialog.status.text())
            dialog.close();parent.close()

    def test_capture_edits_invalidate_result_but_target_selection_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,dialog=self.dialog(folder)
            dialog.result={'path':Path(folder)/'old.npz'}
            dialog.target.addItem('Other','other');dialog.target.setCurrentIndex(1)
            self.assertIsNotNone(dialog.result)
            dialog.start.setValue(1.)
            self.assertIsNone(dialog.result)
            self.assertFalse(dialog.apply.isEnabled())
            dialog.result={'path':Path(folder)/'old.npz'}
            dialog.predict.setChecked(False)
            self.assertIsNone(dialog.result)
            dialog.close();parent.close()

    def test_saved_trim_uses_local_decode_time_and_global_animation_time(self):
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'saved.mp4'; video.touch()
            window=SimpleNamespace(video_path=video,info=True,workspace_root=Path(folder),project=None,
                                   processing_range=lambda:(12.,15.))
            with patch('mocap.project_workspace.read',return_value={'time_offset':10.}):
                self.assertEqual(capture_source(window),(video,2.,5.,10.))

    def test_missing_original_uses_prepared_clip_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder); video=project/'input.mp4'; video.touch()
            (project/'project.json').write_text(json.dumps({'start':30.}),encoding='utf8')
            window=SimpleNamespace(video_path=project/'missing.mp4',info=True,workspace_root=None,project=project,
                                   processing_range=lambda:(31.,33.))
            self.assertEqual(capture_source(window),(video,1.,3.,30.))

    def test_dialog_keeps_original_timing_in_worker_request(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); video=base/'video.mp4'; video.touch()
            parent=QWidget()
            parent.video_path=video; parent.info=True; parent.workspace_root=base; parent.project=None
            parent.processing_range=lambda:(12.,15.)
            parent.characters=[dict(id='c1',name='Actor',kind='pmx')]
            parent.selected_character=lambda:parent.characters[0]
            parent.character_path=lambda relative:base/'characters'/relative
            with patch('mocap.project_workspace.read',return_value={'time_offset':10.}), \
                 patch('mocap.face.runtime.status',return_value=dict(ready=True,message='Ready')), \
                 patch('mocap.face.runtime.model_path',return_value=base/'model.task'), \
                 patch('mocap.face.runtime.command',side_effect=lambda p:[str(p)]):
                dialog=FaceCaptureDialog(parent)
                with patch.object(dialog,'start_process') as start:
                    dialog.run_capture()
                request=Path(start.call_args.args[0][0])
                job=json.loads(request.read_text(encoding='utf8'))
                self.assertEqual((job['start_seconds'],job['end_seconds'],job['timeline_offset_seconds']),(2.,5.,10.))
                self.assertTrue(job['mirror'])  # native anatomical left/right, no swap
                self.assertTrue(job['prediction'])
                self.assertTrue(job['smoothing'])
                dialog.predict.setChecked(False)
                with patch.object(dialog,'start_process') as start:
                    dialog.run_capture()
                disabled=json.loads(Path(start.call_args.args[0][0]).read_text(encoding='utf8'))
                self.assertFalse(disabled['prediction'])
                self.assertEqual(dialog.target.currentData(),'c1')
                self.assertFalse(dialog.apply.isEnabled())
                dialog.close(); parent.close()

    def test_vmd_binary_names_weights_and_frame_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'face.npz'; target=Path(folder)/'动作.vmd'
            names=np.array(['まばたき','あ'])
            np.savez(source,morph_names=names,weights=np.array([[0,.25],[1,.75]],np.float32),
                     fps=30.,start_seconds=15.,detected=np.array([True,True]))
            export_vmd(source,target)
            data=target.read_bytes()
            self.assertEqual(data[:25],b'Vocaloid Motion Data 0002')
            self.assertEqual(struct.unpack_from('<II',data,50),(0,4))
            records=[]
            for i in range(4):
                offset=58+i*23
                name=data[offset:offset+15].rstrip(b'\0').decode('cp932')
                frame,weight=struct.unpack_from('<If',data,offset+15)
                records.append((name,frame,weight))
            self.assertEqual(records,[('まばたき',0,0.),('あ',0,.25),('まばたき',1,1.),('あ',1,.75)])
            self.assertEqual(struct.unpack_from('<4I',data,150),(0,0,0,0))
            self.assertEqual(len(data),166)

    def test_invalid_track_does_not_replace_existing_vmd(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'face.npz'; target=Path(folder)/'keep.vmd'
            target.write_bytes(b'original')
            np.savez(source,morph_names=np.array(['あ']),weights=np.array([[float('nan')]]),
                     fps=30.,start_seconds=0.,detected=np.array([True]))
            with self.assertRaises(ValueError):export_vmd(source,target)
            self.assertEqual(target.read_bytes(),b'original')


if __name__=='__main__': unittest.main()

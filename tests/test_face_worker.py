"""Transactional face application and session-to-project copy regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.retarget import worker


class FaceWorkerTest(unittest.TestCase):
    def test_face_failure_preserves_original_and_never_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';take=source/'take-0';take.mkdir(parents=True)
            (take/'import.json').write_text('{"kind":"pmx"}',encoding='utf-8')
            (take/'animation.blend').write_bytes(b'original body animation')
            face=root/'face.npz';np.savez(face,morph_names=['あ'],weights=np.zeros((2,1),np.float32),fps=30.,start_seconds=4.,detected=np.ones(2,bool))
            job=dict(mode='face',output=str(root/'output'),asset=str(take),source_animation=str(source),
                     takes=[dict(folder='take-0',start=4.,frames=2,fps=30.)],face_motion=str(face))
            request=root/'request.json';request.write_text(json.dumps(job),encoding='utf-8')
            def fail(mode,out,*args):
                (out/'animation.blend').write_bytes(b'partial new output')
                raise RuntimeError('unsupported shape keys')
            with patch.object(worker,'invoke',fail),self.assertRaises(RuntimeError):worker.main(['--request',str(request)])
            self.assertEqual((take/'animation.blend').read_bytes(),b'original body animation')
            self.assertFalse((root/'output/complete.json').exists())

    def test_adoption_copies_face_and_animated_takes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'session';(source/'person/asset').mkdir(parents=True)
            (source/'person/asset/import.json').write_text('{"kind":"pmx"}',encoding='utf-8')
            (source/'person/animation/take-0').mkdir(parents=True)
            (source/'person/animation/take-0/animation.blend').write_bytes(b'body and face')
            (source/'face.npz').write_bytes(b'face capture')
            character=dict(id='person',name='Test',asset='person/asset',animation='person/animation',face_motion='face.npz',takes=[dict(folder='take-0',start=4.,frames=10,fps=30.)])
            request=root/'request.json';request.write_text(json.dumps(dict(mode='adopt',output=str(root/'output'),source=str(source),prefix='copied',characters=[character])),encoding='utf-8')
            worker.main(['--request',str(request)])
            saved=json.loads((root/'output/complete.json').read_text(encoding='utf-8'))['characters'][0]
            self.assertEqual(saved['animation'],'copied/person/animation')
            self.assertEqual(saved['face_motion'],'copied/person/face.npz')
            self.assertEqual((root/'output/person/animation/take-0/animation.blend').read_bytes(),b'body and face')
            self.assertEqual((source/'face.npz').read_bytes(),b'face capture')


if __name__=='__main__':unittest.main()

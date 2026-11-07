"""Readiness and launch guards; no models or network required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mocap.face import runtime


class FaceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='face-runtime-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(runtime, 'ROOT', self.root).start()
        (self.root / 'installer').mkdir()
        (self.root / 'installer/face-runtime-lock.json').write_text('{}', encoding='utf8')
        (self.root / '.setup').mkdir()
        self.folder = self.root / 'runtime_face/test-generation'
        (self.folder / 'Scripts').mkdir(parents=True)
        (self.folder / 'Scripts/python.exe').write_bytes(b'python')
        self.model = self.folder / 'face_landmarker.task'
        self.model.write_bytes(b'model')
        self.receipt = dict(schema=1, identity=runtime.identity(),
                            folder='runtime_face/test-generation',
                            verified_files={'face_landmarker.task':
                                            [self.model.stat().st_size, self.model.stat().st_mtime_ns]})
        self.save()

    def save(self):
        (self.root / '.setup/face-runtime.json').write_text(json.dumps(self.receipt), encoding='utf8')

    def test_verified_launch_keeps_unicode_request_as_one_argument(self):
        self.assertTrue(runtime.status()['ready'])
        request = self.root / '日本語 路径/face request.json'
        command = runtime.command(request)
        self.assertEqual(command[-2:], ['--request', str(request.resolve())])
        self.assertEqual(command[1:4], ['-I', '-X', 'utf8'])
        self.assertEqual(runtime.model_path(), self.model)

    def test_model_replacement_invalidates_readiness(self):
        self.model.write_bytes(b'changed model')
        self.assertFalse(runtime.status()['ready'])
        with self.assertRaises(ValueError):
            runtime.command(self.root / 'request.json')

    def test_new_dependencies_require_setup(self):
        (self.root / 'installer/face-runtime-lock.json').write_text('{"schema":2}', encoding='utf8')
        self.assertFalse(runtime.status()['ready'])

    def test_receipt_cannot_escape_runtime_generation(self):
        self.receipt['folder'] = 'runtime_face/../../elsewhere'
        self.save()
        self.assertFalse(runtime.status()['ready'])

    def test_watched_file_cannot_escape_generation(self):
        self.receipt['verified_files']['../../../installer/face-runtime-lock.json'] = [2, 0]
        self.save()
        self.assertFalse(runtime.status()['ready'])

    def test_missing_receipt_is_actionable(self):
        (self.root / '.setup/face-runtime.json').unlink()
        status = runtime.status()
        self.assertFalse(status['ready'])
        self.assertIn('Setup_Face.cmd', status['message'])


if __name__ == '__main__':
    unittest.main()

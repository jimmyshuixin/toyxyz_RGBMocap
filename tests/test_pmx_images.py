"""Missing optional maps must not discard an otherwise usable PMX import."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PmxImagePackingTests(unittest.TestCase):
    def test_missing_and_unreadable_maps_are_reported_and_valid_map_is_packed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            valid = folder / 'body.png'
            valid.write_bytes(b'fixture')
            broken = folder / 'broken.png'
            broken.write_bytes(b'fixture')
            def image(name, path, size=(2, 2), packed=False):
                return SimpleNamespace(name=name, filepath=str(path), size=size,
                                       source='FILE', packed_file=packed, library=None, pack=Mock())
            body = image('body', valid)
            toon = image('toon01', folder / 'toon01.bmp', (0, 0))
            unreadable = image('broken', broken, (0, 0))
            packed = image('packed', folder / 'no-longer-present.png', packed=True)
            fake = SimpleNamespace(data=SimpleNamespace(images=[body, toon, unreadable, packed]),
                                   path=SimpleNamespace(abspath=lambda path, **_: path))
            spec = importlib.util.spec_from_file_location('mocap.retarget._image_test', ROOT / 'mocap/retarget/blender_images.py')
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {'bpy': fake}):
                spec.loader.exec_module(module)
                warnings = module.pack_images()
            self.assertEqual([row['code'] for row in warnings], ['missing_texture', 'unreadable_texture'])
            body.pack.assert_called_once_with()
            for missing in (toon, unreadable, packed):
                missing.pack.assert_not_called()
            self.assertEqual(valid.read_bytes(), b'fixture')
            self.assertEqual(broken.read_bytes(), b'fixture')
            self.assertFalse((folder / 'toon01.bmp').exists())


if __name__ == '__main__':
    unittest.main()

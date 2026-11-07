"""Original PMX helper whitelist used by preview and FBX sampling."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
module = ModuleType('io_scene_fbx')
module.export_fbx_bin = NS()
spec = importlib.util.spec_from_file_location('_pmx_dependencies', ROOT / 'mocap/retarget/selective_fbx.py')
dependencies = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'io_scene_fbx': module}):
    spec.loader.exec_module(dependencies)


class PmxDependencyTests(unittest.TestCase):
    def test_existing_standard_helpers_are_selected_without_extra_controls(self):
        bones = {'足D.L', '足D.R', 'ひざ補助.L', '胸補助.R', 'arbitrary_cuff', 'heel_IK', 'root'}
        arm = NS(data=NS(bones=bones))
        self.assertEqual(dependencies.body_dependencies('pmx', arm),
                         {'足D.L', '足D.R', 'ひざ補助.L', '胸補助.R'})
        self.assertEqual(arm.data.bones, bones)

    def test_missing_helper_bones_are_not_invented(self):
        self.assertEqual(dependencies.body_dependencies('pmx', NS(data=NS(bones={'root'}))), set())

    def test_other_character_kinds_do_not_use_pmx_helpers(self):
        for kind in ('vrm', 'mixamo'):
            self.assertEqual(dependencies.body_dependencies(kind, NS(data=NS(bones={'足D.L'}))), set())


if __name__ == '__main__':
    unittest.main()

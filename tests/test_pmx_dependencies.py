"""Keep native PMX append-driven skin deformation in preview/FBX sampling."""
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


def rig():
    bones, poses = {}, []
    def add(name, parent=None, grant=None, mute=False, shadow=False):
        bones[name] = NS(name=name, parent=bones.get(parent))
        poses.append(NS(name=name, bone=bones[name], mmd_bone=NS(additional_transform_bone=grant or ''),
                        is_mmd_shadow_bone=shadow,
                        constraints=[] if grant is None else [NS(name='mmd_additional_rotation', mute=mute, influence=1.)]))
    add('root')
    add('forearm', 'root')
    add('wrist', 'forearm')
    add('arbitrary_cuff', 'forearm', 'wrist')
    add('cuff_child', 'arbitrary_cuff')
    add('chained_helper', None, 'cuff_child')
    add('muted_helper', 'forearm', 'wrist', mute=True)
    add('unrelated_control')
    add('unrelated_helper', None, 'unrelated_control')
    add('_shadow', 'forearm', 'wrist', shadow=True)
    return NS(data=NS(bones=list(bones.values())), pose=NS(bones=poses))


class PmxDependencyTests(unittest.TestCase):
    def test_native_grant_closure_keeps_arbitrary_helper_names(self):
        selected = dependencies.body_dependencies('pmx', rig(), {'forearm', 'wrist'})
        self.assertEqual(selected, {'arbitrary_cuff', 'chained_helper'})

    def test_unrelated_static_controls_are_not_added(self):
        self.assertEqual(dependencies.body_dependencies('pmx', rig(), ()), set())

    def test_active_accessory_ik_keeps_its_whole_affected_chain(self):
        arm = rig()
        cuff = arm.pose.bones[3]
        cuff.constraints = [NS(name='IK', type='IK', mute=False, influence=1.,
                               subtarget='wrist', chain_count=2)]
        self.assertEqual(dependencies.body_dependencies('pmx', arm, {'forearm'}),
                         {'forearm', 'arbitrary_cuff', 'chained_helper'})
        cuff.constraints[0].mute = True
        self.assertEqual(dependencies.body_dependencies('pmx', arm, {'wrist'}), set())

    def test_cycles_terminate_without_mutating_the_rig(self):
        arm = rig()
        a, b = arm.pose.bones[3], arm.pose.bones[5]
        a.mmd_bone.additional_transform_bone = b.name
        b.mmd_bone.additional_transform_bone = a.name
        self.assertEqual(dependencies.body_dependencies('pmx', arm, {a.name}), {a.name, b.name})
        self.assertEqual(a.mmd_bone.additional_transform_bone, b.name)

    def test_other_character_kinds_keep_previous_behavior(self):
        for kind in ('vrm', 'mixamo'):
            self.assertEqual(dependencies.body_dependencies(kind, rig(), {'wrist'}), set())


if __name__ == '__main__':
    unittest.main()

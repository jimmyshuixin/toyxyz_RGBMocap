"""Fail explicitly instead of publishing wrong PMX auxiliary-parent motion."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from test_longvideo_retarget import bake_fixture


class PmxAncestorControlTests(unittest.TestCase):
    def test_active_grant_parent_is_rejected_before_animation_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, keyed, *_ = bake_fixture(Path(temporary), 5)
            arm = bake.bpy.data.objects['Armature']
            hand = arm.data.bones['手首.L']
            helper = NS(name='ひじC.L', parent=hand.parent)
            arm.data.bones.append(helper)
            hand.parent = helper
            arm.pose.bones.append(NS(name=helper.name, mmd_bone=NS(additional_transform_bone=hand.parent.parent.name), constraints=[
                NS(name='mmd_additional_rotation', mute=False, influence=1.)]))
            with self.assertRaisesRegex(ValueError, 'active append-transform ancestors: ひじC.L'):
                bake.validate_pmx_ancestor_controls(arm, {'手首.L', helper.parent.name})
            self.assertEqual(keyed, {})
            self.assertFalse((Path(temporary) / 'animation.blend').exists())

    def test_muted_and_explicit_waist_cancel_controls_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, *_ = bake_fixture(Path(temporary), 5)
            arm = bake.bpy.data.objects['Armature']
            hand = arm.data.bones['手首.L']
            for name, mute in [('custom_parent', True), ('腰キャンセル.L', False)]:
                helper = NS(name=name, parent=hand.parent)
                arm.data.bones.append(helper)
                hand.parent = helper
                arm.pose.bones.append(NS(name=name, mmd_bone=NS(additional_transform_bone='手首.L'), constraints=[
                    NS(name='mmd_additional_rotation', mute=mute, influence=1.)]))
            bake.validate_pmx_ancestor_controls(arm, {'手首.L'})

    def test_static_shoulder_cancel_parent_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, *_ = bake_fixture(Path(temporary), 5)
            arm = bake.bpy.data.objects['Armature']
            hand = arm.data.bones['手首.L']
            helper = NS(name='肩C.L', parent=hand.parent)
            arm.data.bones.append(helper)
            hand.parent = helper
            arm.pose.bones.append(NS(name=helper.name,
                mmd_bone=NS(additional_transform_bone='static_shoulder_P'), constraints=[
                    NS(name='mmd_additional_rotation', mute=False, influence=1.)]))
            bake.validate_pmx_ancestor_controls(arm, {'手首.L', helper.parent.name})

    def test_transitive_local_grant_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, *_ = bake_fixture(Path(temporary), 5)
            arm = bake.bpy.data.objects['Armature']
            hand = arm.data.bones['手首.L']
            helper = NS(name='ひじC.L', parent=hand.parent)
            arm.data.bones.append(helper)
            hand.parent = helper
            constraint = NS(name='mmd_additional_rotation', mute=False, influence=1.)
            arm.pose.bones.append(NS(name=helper.name,
                mmd_bone=NS(additional_transform_bone='ひじ中継.L'), constraints=[constraint]))
            arm.pose.bones.append(NS(name='ひじ中継.L',
                mmd_bone=NS(additional_transform_bone=helper.parent.name), constraints=[constraint]))
            with self.assertRaisesRegex(ValueError, 'active append-transform ancestors: ひじC.L'):
                bake.validate_pmx_ancestor_controls(arm, {'手首.L', helper.parent.name})

    def test_independent_ik_driven_grant_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, *_ = bake_fixture(Path(temporary), 5)
            arm = bake.bpy.data.objects['Armature']
            hand = arm.data.bones['手首.L']
            helper = NS(name='ひじC.L', parent=hand.parent)
            driver = NS(name='duplicate_elbow', parent=None)
            arm.data.bones.extend([helper, driver])
            hand.parent = helper
            arm.pose.bones.append(NS(name=helper.name,
                mmd_bone=NS(additional_transform_bone=driver.name), constraints=[
                    NS(name='mmd_additional_rotation', mute=False, influence=1.)]))
            arm.pose.bones.append(NS(name=driver.name, bone=driver,
                constraints=[NS(name='IK', type='IK', mute=False, influence=1., chain_count=1)]))
            with self.assertRaisesRegex(ValueError, 'active append-transform ancestors: ひじC.L'):
                bake.validate_pmx_ancestor_controls(arm, {'手首.L'})


if __name__ == '__main__':
    unittest.main()

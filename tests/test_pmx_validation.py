from types import SimpleNamespace
import unittest

from mocap.retarget.pmx_validation import self_grants, validate_bones


class PmxDependencyTests(unittest.TestCase):
    def test_active_self_grants_rejected_with_names(self):
        bone = SimpleNamespace(name='wing', hasAdditionalRotate=True,
                               hasAdditionalLocation=False, additionalTransform=(0, 1.))
        with self.assertRaisesRegex(ValueError, r'wing \(#0\)'):
            validate_bones([bone])

    def test_zero_influence_and_external_grants_remain_supported(self):
        bones = [SimpleNamespace(name='root', hasAdditionalRotate=True, hasAdditionalLocation=False,
                                 additionalTransform=(0, 0.)),
                 SimpleNamespace(name='cuff', hasAdditionalRotate=True, hasAdditionalLocation=False,
                                 additionalTransform=(0, -.5))]
        self.assertEqual(self_grants(bones), [])
        validate_bones(bones)


if __name__ == '__main__':
    unittest.main()

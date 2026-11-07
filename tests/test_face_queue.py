"""Checkbox queue contracts, independent of models, Qt and user projects."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.face.queue import (SUPPORTED_MODELS, append_capture_jobs,
                              face_arguments, split_people_stages)


class FaceQueueTests(unittest.TestCase):
    def test_enabled_reconstruction_appends_after_body_export(self):
        jobs = [('prepare', ['--video', 'clip.mp4']), ('track', None),
                ('exmr', None), ('correct', ['--result', 'corrected_initial']),
                ('export', ['--result', 'corrected_initial'])]
        original = list(jobs)
        options = {'enabled': True, 'smoothing': False}
        result = append_capture_jobs(jobs, model='exmr', options=options)
        self.assertEqual(jobs, original)
        self.assertEqual(result[:-1], original)
        self.assertEqual(result[-1][0], 'face_capture')
        self.assertNotIn('--project', result[-1][1])
        options['smoothing'] = True
        self.assertEqual(json.loads(result[-1][1][1]), {'enabled': True, 'smoothing': False})

    def test_disabled_and_unsupported_modes_keep_queue_unchanged(self):
        jobs = [('export', ['--project', 'actor'])]
        for model in SUPPORTED_MODELS:
            self.assertEqual(append_capture_jobs(jobs, model=model, options={}), jobs)
            self.assertEqual(append_capture_jobs(jobs, model=model, options={'enabled': False}), jobs)
        for model in ('mamma', 'prompthmr', 'pear'):
            self.assertEqual(append_capture_jobs(jobs, model=model, options={'enabled': True}), jobs)

    def test_people_targets_are_explicit_and_deduplicated(self):
        projects = ['scene/person_0001/take-0', 'scene/person_0002/take-0',
                    'scene/person_0001/take-1', 'scene/person_0002/take-0']
        jobs = [('flow_batch', ['--batch-items', 'preserve']), ('flow_export', ['--project', projects[0]])]
        result = append_capture_jobs(jobs, model='flowhmr', options={'enabled': True}, projects=projects)
        self.assertEqual(result[:2], jobs)
        self.assertEqual([args[1] for _, args in result[2:]], [str(Path(path)) for path in projects[:3]])
        self.assertTrue(all(stage == 'face_capture' for stage, _ in result[2:]))
        again = append_capture_jobs(result, model='flowhmr', options={'enabled': True}, projects=projects)
        self.assertEqual(again, result)

    def test_no_selected_targets_never_captures_the_active_actor_implicitly(self):
        jobs = [('gvhmr', None)]
        self.assertEqual(append_capture_jobs(jobs, model='gvhmr', options={'enabled': True}, projects=[]), jobs)

    def test_person_faces_are_deferred_until_after_contacts_and_export(self):
        stages = ['flow_features', 'face_capture', 'flowhmr', 'correct', 'flow_export']
        inference, post = split_people_stages(stages)
        self.assertEqual(inference, ['flow_features', 'flowhmr'])
        self.assertEqual(post, ['correct', 'flow_export', 'face_capture'])
        self.assertEqual(stages[1], 'face_capture')  # caller's data was not reordered

    def test_existing_nonface_people_order_is_preserved(self):
        self.assertEqual(split_people_stages(['pva', 'gvhmr', 'refine', 'correct', 'export']),
                         (['pva', 'gvhmr', 'refine'], ['correct', 'export']))

    def test_unicode_paths_and_strict_boolean_options(self):
        path = Path('人物/镜头 1')
        args = face_arguments({'enabled': True}, path)
        self.assertEqual(args[:2], ['--project', str(path)])
        self.assertEqual(json.loads(args[-1]), {'enabled': True, 'smoothing': True})
        for options in ({'enabled': 'false'}, {'enabled': 1}, {'smoothing': None}):
            with self.assertRaises(ValueError): face_arguments(options)


if __name__ == '__main__':
    unittest.main()

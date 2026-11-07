"""CPU checks for HTD's optimization-to-IK gradient boundary.

The real pipeline, FK, CCD IK, optimizer and loss code run with a deterministic
synthetic body model. No licensed geometry, weights, CUDA or video is required.
The reference retains the old grad-enabled IK call before detaching its output.
"""
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import importlib
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
HTD = ROOT / 'third_party' / 'HTD-Refine' / 'optimization'
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


@contextmanager
def isolated_modules(package, body_module):
    prefix = package.__name__
    def owned(name):
        return name == prefix or name.startswith(prefix + '.')
    previous = {name: value for name, value in sys.modules.items() if owned(name)}
    sys.modules.update({prefix: package, body_module.__name__: body_module})
    try:
        yield
    finally:
        # Do not unload unrelated native imports such as cv2. Reimporting a
        # native extension after restoring the entire sys.modules is unsafe.
        for name in tuple(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(previous)


@contextmanager
def synthetic_pipeline():
    """Isolate upstream's generic package name and replace only body assets."""
    name = '_memory_htd_test'
    package = ModuleType(name)
    package.__path__ = [str(HTD)]
    body_module = ModuleType(name + '.body_model')
    with isolated_modules(package, body_module), \
            patch.object(sys, 'path', [str(ROOT / 'native_compat'), *sys.path]):
        matrix = importlib.import_module(name + '.matrix')
        from pytorch3d.transforms import axis_angle_to_matrix

        class SyntheticBody(torch.nn.Module):
            def __init__(self):
                super().__init__()
                generator = torch.Generator().manual_seed(231)
                offsets = torch.randn(22, 3, generator=generator) * .12
                offsets[0] = 0
                rest = offsets.clone()
                for joint, parent in enumerate(PARENTS):
                    if parent >= 0:
                        rest[joint] += rest[parent]
                self.register_buffer('parents', torch.tensor(PARENTS))
                self.register_buffer('rest', rest)

            def get_skeleton(self, betas):
                return self.rest.expand(*betas.shape[:-1], 22, 3)

            def forward(self, body_pose, betas, global_orient, transl):
                shape = body_pose.shape[:-1]
                rotations = axis_angle_to_matrix(
                    torch.cat((global_orient, body_pose), -1).reshape(*shape, 22, 3))
                rest = self.get_skeleton(betas)
                local = rest - rest[..., self.parents, :]
                local = torch.cat((rest[..., :1, :] + transl[..., None, :], local[..., 1:, :]), -2)
                transforms = matrix.get_TRS(rotations, local)
                world = matrix.forward_kinematics(transforms, PARENTS)
                return matrix.get_position(world)[..., :17, :]

        body_module.create_body_model = lambda *args, **kwargs: SyntheticBody()
        pipeline = importlib.import_module(name + '.pipeline')
        yield pipeline


def fixture(frames):
    generator = torch.Generator().manual_seed(715 + frames)
    params = {
        'body_pose': torch.randn(frames, 63, generator=generator) * .04,
        'global_orient': torch.randn(frames, 3, generator=generator) * .02,
        'transl': torch.randn(frames, 3, generator=generator) * .002 + torch.tensor([0., 0., 3.]),
        'betas': torch.zeros(frames, 10),
    }
    intrinsics = torch.tensor([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    return dict(
        tgt_vel_3d=torch.randn(frames - 1, 17, 3, generator=generator) * .04,
        tgt_accel_3d=torch.zeros(frames - 2, 17, 3),
        tgt_kp2d=torch.zeros(frames, 17, 2),
        K=intrinsics.expand(frames, 3, 3),
        T_w2c_tgt=torch.eye(4).expand(frames, 4, 4),
        pred_smpl_params_incam=params,
        pred_smpl_params_global=params,
        model='smplx', static=True, preserve_input_camera=True,
    )


def trainable_copy(params):
    return {key: value.detach().clone().requires_grad_(key != 'betas') for key, value in params.items()}


class SavedTensors:
    def __init__(self):
        self.count = 0

    def pack(self, value):
        self.count += 1
        return value

    def __enter__(self):
        self.context = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda value: value)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)


class HTDMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.scope = synthetic_pipeline()
        cls.pipeline = cls.scope.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.scope.__exit__(None, None, None)
        torch.set_num_threads(cls.threads)

    def test_ik_matches_legacy_boundary_without_saved_tensors(self):
        pipeline = self.pipeline
        original_ik = pipeline._apply_ik_postprocess
        for frames in (4, 60, 240):
            with self.subTest(frames=frames), redirect_stdout(io.StringIO()):
                args = fixture(frames)
                optimized = trainable_copy(args['pred_smpl_params_global'])
                with SavedTensors() as legacy_saved:
                    expected = original_ik(trainable_copy(optimized), args['tgt_vel_3d'], 'smplx')
                expected = {key: value.detach().clone() for key, value in expected.items()}
                self.assertGreater(legacy_saved.count, 0)
                observations = []

                def observe_ik(params, *positional, **kwargs):
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertTrue(all(not value.requires_grad for value in params.values()))
                    with SavedTensors() as saved:
                        result = original_ik(params, *positional, **kwargs)
                    observations.append(saved.count)
                    return result

                with patch.object(pipeline, 'optimize_smpl', return_value=optimized), \
                        patch.object(pipeline, '_apply_ik_postprocess', side_effect=observe_ik):
                    actual, camera, scale = pipeline.opt_vel(**args)
                self.assertEqual(observations, [0])
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                    self.assertFalse(actual[key].requires_grad, key)
                torch.testing.assert_close(camera, args['T_w2c_tgt'], rtol=0, atol=1e-6)
                self.assertEqual(scale.item(), 1.)

    def test_optimization_still_builds_gradients_and_updates_pose(self):
        pipeline = self.pipeline
        args = fixture(8)
        # Exercise the actual optimizer/closure, with two iterations instead of
        # a full fitting job. Only the licensed body model is a stand-in.
        stage = replace(pipeline.OPTIMIZATION_CONFIG.motion_refinement,
                        max_steps=2, milestones=(3,), warmup=1)
        config = replace(pipeline.OPTIMIZATION_CONFIG, motion_refinement=stage)
        original_optimizer = pipeline.optimize_smpl
        observations = []

        def observe_optimizer(**kwargs):
            self.assertTrue(torch.is_grad_enabled())
            with SavedTensors() as saved:
                result = original_optimizer(**kwargs)
            self.assertGreater(saved.count, 0)
            self.assertTrue(all(result[key].requires_grad for key in stage.fit_keys))
            self.assertTrue(any(result[key].grad is not None and
                                torch.count_nonzero(result[key].grad).item() > 0
                                for key in stage.fit_keys))
            self.assertFalse(torch.equal(result['body_pose'], kwargs['smpl_params']['body_pose']))
            observations.append(True)
            return result

        with patch.object(pipeline, 'OPTIMIZATION_CONFIG', config), \
                patch.object(pipeline, 'optimize_smpl', side_effect=observe_optimizer), \
                redirect_stdout(io.StringIO()):
            result, _, _ = pipeline.opt_vel(**args)
        self.assertEqual(observations, [True])
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))

    def test_disabled_ik_keeps_values_and_detaches_optimizer_output(self):
        pipeline = self.pipeline
        args = fixture(8)
        optimized = trainable_copy(args['pred_smpl_params_global'])
        with patch.object(pipeline, 'optimize_smpl', return_value=optimized), \
                patch.object(pipeline, '_apply_ik_postprocess') as ik, \
                redirect_stdout(io.StringIO()):
            result, _, _ = pipeline.opt_vel(**args, pp=False)
        ik.assert_not_called()
        for key in optimized:
            self.assertTrue(torch.equal(result[key], optimized[key]), key)
            self.assertFalse(result[key].requires_grad, key)


if __name__ == '__main__':
    unittest.main()

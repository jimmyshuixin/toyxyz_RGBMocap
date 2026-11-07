"""CPU regressions for bounded SimpleVO pixels and model-owned rotary tables.

Uses synthetic images/tensors, real SIFT/FLANN/COLMAP, and frozen pre-change
oracles. Does not exercise a licensed model, CUDA inference or pose accuracy.
"""

import gc
from pathlib import Path
import sys
import unittest
from unittest import mock
import weakref

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ExMR"))
from exmr.model import layers
from exmr.preprocessing import visual_odometry as vo


def original_poses(model, frame_count):
    """Frozen eager-image path before the streaming change, same native solvers."""
    vo._reset_random_state()
    sample_idxs = sorted({*range(0, frame_count, model.step), frame_count - 1})
    frames = [
        cv2.cvtColor(
            cv2.resize(image, None, fx=model.scale, fy=model.scale,
                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB,
        )
        for image in model.frame_reader(sample_idxs)
    ]
    height, width = frames[0].shape[:2]
    focal = (width**2 + height**2)**0.5 / (24**2 + 36**2)**0.5 * model.f_mm
    camera = vo.pycolmap.Camera(camera_id=0, model="SIMPLE_PINHOLE", width=width,
                               height=height, params=[focal, width / 2, height / 2])
    options = vo.pycolmap.TwoViewGeometryOptions(
        min_num_inliers=10, min_E_F_inlier_ratio=0.8,
        max_H_inlier_ratio=0.9, compute_relative_pose=True,
    )
    poses = [np.eye(4)]
    features = model._features(frames[0])
    for frame in frames[1:]:
        previous, features = features, model._features(frame)
        points0, points1 = vo._match(previous, features)
        matches = np.stack([np.arange(len(points0)), np.arange(len(points0))], axis=-1)
        answer = vo.pycolmap.estimate_calibrated_two_view_geometry(
            camera, points0.astype(np.float64), camera, points1.astype(np.float64),
            matches=matches, options=options,
        )
        delta = np.eye(4)
        delta[:3] = answer.cam2_from_cam1.matrix().astype(np.float32)
        poses.append(delta @ poses[-1])
    return vo._interpolate_poses(poses, sample_idxs)


class OriginalRoPE(nn.Module):
    """Frozen per-layer implementation, including its cache growth behavior."""

    def __init__(self, width, max_length=4096):
        super().__init__()
        self.width, self.max_length = width, max_length
        self.register_buffer("angles", layers._rope_angles(width, max_length), False)
        self.cache = None

    def forward(self, x):
        length = x.shape[-2]
        cache = self.cache
        if (cache is None or cache[0].shape[0] < length
                or cache[0].device != x.device or cache[0].dtype != x.dtype):
            angles = (self.angles if length <= self.max_length else
                      layers._rope_angles(self.width, length).to(x.device))
            self.cache = (angles.cos().to(x.dtype), angles.sin().to(x.dtype))
        cos, sin = (value[:length] for value in self.cache)
        split = x.unflatten(-1, (-1, 2))
        first, second = split.unbind(dim=-1)
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        return (x * cos) + (rotated * sin)


def rotary_model():
    model = nn.ModuleDict({
        "body": nn.ModuleList([layers.RoPE(64), layers.RoPE(64)]),
        "foot": nn.ModuleList([layers.RoPE(64)]),
        "hand": nn.ModuleList([layers.RoPE(128), layers.RoPE(128)]),
    })
    layers.share_rope_cache(model)
    return model


class SimpleVOStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(75)
        cls.texture = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    def reader(self, indices):
        for index in indices:
            yield cv2.warpAffine(
                self.texture,
                np.array([[1, 0, index * .5], [0, 1, index * .1]], dtype=np.float32),
                (320, 240), borderMode=cv2.BORDER_REFLECT,
            )

    def test_native_camera_poses_match_eager_oracle_exactly(self):
        for scale, count in ((1., 1), (1., 33), (.5, 33), (1., 34)):
            with self.subTest(scale=scale, frames=count):
                before = vo.SimpleVO(scale=scale, progress=False, frame_reader=self.reader)
                after = vo.SimpleVO(scale=scale, progress=False, frame_reader=self.reader)
                expected = original_poses(before, count)
                actual = after.poses(frame_count=count)
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_array_equal(actual, expected)

    def test_decoded_rgb_residency_is_bounded_and_reader_closes(self):
        decoded, converted, closed = [], [], []
        first_feature_read_count = []

        def reader(indices):
            try:
                for index, image in zip(indices, self.reader(indices)):
                    decoded.append(index)
                    yield image
            finally:
                closed.append(True)

        convert = cv2.cvtColor
        peak = 0

        def count_rgb(*args, **kwargs):
            nonlocal peak
            image = convert(*args, **kwargs)
            if args[1] == cv2.COLOR_BGR2RGB:
                converted.append(weakref.ref(image))
                peak = max(peak, sum(ref() is not None for ref in converted))
            return image

        model = vo.SimpleVO(scale=1., progress=False, frame_reader=reader)
        features = model._features

        def observe_features(image):
            first_feature_read_count.append(len(decoded))
            return features(image)

        # Plain callables avoid Mock's call-history retaining every image.
        with mock.patch.object(vo.cv2, "cvtColor", new=count_rgb), \
                mock.patch.object(model, "_features", new=observe_features):
            model.poses(frame_count=129)
        self.assertEqual(decoded, list(range(0, 129, 8)))
        self.assertEqual(first_feature_read_count[0], 1)
        self.assertLessEqual(peak, 2)
        self.assertEqual(closed, [True])
        self.assertTrue(all(ref() is None for ref in converted))

    def test_reader_closes_on_solver_failure_and_cancellation(self):
        for failure in (RuntimeError("injected failure"), KeyboardInterrupt()):
            for operation in ("_features", "_match", "_interpolate_poses"):
                with self.subTest(operation=operation, error=type(failure).__name__):
                    closed = []

                    def reader(indices):
                        try:
                            yield from self.reader(indices)
                        finally:
                            closed.append(True)

                    model = vo.SimpleVO(scale=1., progress=False, frame_reader=reader)
                    target = model if operation == "_features" else vo
                    with mock.patch.object(target, operation, side_effect=failure):
                        with self.assertRaises(type(failure)):
                            model.poses(frame_count=33)
                    self.assertEqual(closed, [True])

    def test_reader_failure_propagates_and_releases_reader(self):
        closed = []

        def reader(indices):
            try:
                yield next(self.reader(indices))
                raise OSError("injected decode failure")
            finally:
                closed.append(True)

        model = vo.SimpleVO(scale=1., progress=False, frame_reader=reader)
        with self.assertRaisesRegex(OSError, "decode failure"):
            model.poses(frame_count=33)
        self.assertEqual(closed, [True])


class SharedRoPETests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(20261107)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_exact_against_original_at_all_table_boundaries_and_dtypes(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                model = rotary_model()
                references = {width: OriginalRoPE(width) for width in (64, 128)}
                for length in (1, 256, 4096, 4097, 10800, 3):
                    for name in ("body", "foot", "hand"):
                        for rope in model[name]:
                            x = torch.randn(1, 1, length, rope.d_model, dtype=dtype)
                            actual = rope(x)
                            expected = references[rope.d_model](x)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_two_shared_tables_grow_without_retaining_obsolete_copies(self):
        model = rotary_model()
        with torch.inference_mode():
            for name in model:
                for rope in model[name]:
                    rope(torch.zeros(1, 1, 5000, rope.d_model))
            tables = [rope._cos_sin_cache for group in model.values() for rope in group]
            self.assertEqual(len({table[0].data_ptr() for table in tables}), 2)
            old = weakref.ref(model["body"][0]._cos_sin_cache[0])
            del tables
            model["foot"][0](torch.zeros(1, 1, 6000, 64))
            self.assertIsNone(old())
            self.assertIs(model["body"][0]._cos_sin_cache[0],
                          model["foot"][0]._cos_sin_cache[0])
            model["body"][0](torch.zeros(1, 1, 2, 64))
            self.assertEqual(model["body"][0]._cos_sin_cache[0].shape[0], 6000)
            cache = model["body"][0]._rope_cache
            self.assertEqual(len(cache.tables), 2)
            actual_bytes = sum(t.numel() * t.element_size()
                               for _, pair in cache.tables.values() for t in pair)
            self.assertEqual(actual_bytes, (6000 * 64 + 5000 * 128) * 2 * 4)

    def test_converted_angle_buffers_preserve_each_layers_mixed_length_history(self):
        for dtype in (torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                model = nn.ModuleList([layers.RoPE(8, max_seq_len=8) for _ in range(2)])
                layers.share_rope_cache(model)
                model.to(dtype=dtype)
                originals = [OriginalRoPE(8, max_length=8).to(dtype=dtype) for _ in range(2)]
                # A long table uses newly generated FP32 angles; B's first
                # short call must instead use its own converted angle buffer.
                for index, length in ((0, 9), (1, 4), (0, 4), (1, 9), (1, 4)):
                    x = torch.randn(1, 1, length, 8, dtype=dtype)
                    torch.testing.assert_close(model[index](x), originals[index](x),
                                               rtol=0, atol=0)
                self.assertIsNot(model[0]._cos_sin_cache[0], model[1]._cos_sin_cache[0])
                old = [weakref.ref(rope._cos_sin_cache[0]) for rope in model]
                model.float()
                self.assertTrue(all(ref() is None for ref in old))
                for rope in model:
                    rope(torch.zeros(1, 1, 9, 8))
                self.assertIsNot(model[0]._cos_sin_cache[0], model[1]._cos_sin_cache[0])

    def test_dtype_roundtrip_preserves_private_mixed_length_histories(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype), torch.inference_mode():
                model = nn.ModuleList([layers.RoPE(8, max_seq_len=8) for _ in range(2)])
                layers.share_rope_cache(model)
                model.to(dtype=dtype).float()
                originals = [OriginalRoPE(8, max_length=8).to(dtype=dtype).float()
                             for _ in range(2)]
                # FP32 dtype does not restore angle values rounded by half or
                # bfloat16. A's long table must not replace B's rounded short one.
                for index, length in ((0, 9), (1, 4), (0, 4), (1, 9), (1, 4)):
                    x = torch.randn(1, 1, length, 8)
                    torch.testing.assert_close(model[index](x), originals[index](x),
                                               rtol=0, atol=0)
                self.assertIsNot(model[0]._cos_sin_cache[0], model[1]._cos_sin_cache[0])

    def test_device_dtype_apply_releases_cached_tensors_and_preserves_math(self):
        model = rotary_model()
        with torch.inference_mode():
            model["body"][0](torch.zeros(1, 1, 5000, 64))
        table = weakref.ref(model["body"][0]._cos_sin_cache[0])
        model.double()
        self.assertIsNone(table())
        self.assertIsNone(model["body"][0]._cos_sin_cache)
        original = OriginalRoPE(64).double()
        with torch.inference_mode():
            x = torch.randn(1, 1, 31, 64, dtype=torch.float64)
            torch.testing.assert_close(model["foot"][0](x), original(x), rtol=0, atol=0)
        model.cpu()
        self.assertIsNone(model["foot"][0]._cos_sin_cache)

    def test_instances_are_isolated_and_destruction_releases_tables(self):
        first, second = rotary_model(), rotary_model()
        with torch.inference_mode():
            x = torch.zeros(1, 1, 5000, 64)
            first["body"][0](x)
            second["body"][0](x)
        self.assertIsNot(first["body"][0]._cos_sin_cache[0],
                         second["body"][0]._cos_sin_cache[0])
        cache = weakref.ref(first["body"][0]._rope_cache)
        table = weakref.ref(first["body"][0]._cos_sin_cache[0])
        del first
        gc.collect()
        self.assertIsNone(cache())
        self.assertIsNone(table())
        self.assertIsNotNone(second["body"][0]._cos_sin_cache)

    def test_encoder_outputs_gradients_and_checkpoint_keys_are_unchanged(self):
        original = layers.RoPEEncoder(32, 2, 4, 2., 0., 120).eval()
        shared = layers.RoPEEncoder(32, 2, 4, 2., 0., 120).eval()
        with torch.no_grad():
            for block in original.blocks:
                block.gate_msa.fill_(.3)
                block.gate_mlp.fill_(.2)
        shared.load_state_dict(original.state_dict(), strict=True)
        keys = list(shared.state_dict())
        layers.share_rope_cache(shared)
        self.assertEqual(list(shared.state_dict()), keys)
        for length in (31, 513):
            for gradients in (False, True):
                # Clear inference-only tensors before the independent training case.
                original.cpu()
                shared.cpu()
                x = torch.randn(1, length, 32, requires_grad=gradients)
                y = x.detach().clone().requires_grad_(gradients)
                with torch.set_grad_enabled(gradients):
                    expected, _ = original(x, torch.tensor([length]))
                    actual, _ = shared(y, torch.tensor([length]))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if gradients:
                        expected.square().sum().backward()
                        actual.square().sum().backward()
                        torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)

    def test_exmr_constructor_shares_across_all_three_networks(self):
        from exmr.model import exmr
        # Instantiate real default temporal networks on meta; licensed geometry
        # and checkpoint files are unnecessary for checking the binding.
        with torch.device("meta"), \
                mock.patch.object(exmr, "Decoder", side_effect=lambda _: nn.Identity()), \
                mock.patch.object(exmr, "SmplxModel", side_effect=lambda _: nn.Identity()):
            model = exmr.ExMR("unused-model-path")
        ropes = [module for module in model.modules() if isinstance(module, layers.RoPE)]
        self.assertEqual(len(ropes), 24)
        self.assertEqual(len({id(rope._rope_cache) for rope in ropes}), 1)
        self.assertEqual({rope.d_model for rope in ropes}, {64, 128})
        self.assertFalse(any("cache" in key or "angles" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()

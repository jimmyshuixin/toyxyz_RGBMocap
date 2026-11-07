"""CPU regression tests for ExMR's bounded, full-sequence inference attention.

Run with the ExMR runtime (or another environment containing PyTorch):
    python tests/test_longvideo_attention.py -v

The oracle below preserves the dense mask/SDPA equations from RGB MoCap commit
ef05127a0dbddb3f3ac672a0722dddeae42ea651. It does not call the new plan builder.
These tests use synthetic tensors and random weights, not licensed checkpoints.
They verify numerical/structural behavior, not mocap accuracy, CUDA memory use,
Windows compatibility, or MMD export.
"""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


EXMR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ExMR"
sys.path.insert(0, str(EXMR_ROOT))
from exmr.model import layers  # noqa: E402
from exmr.model import query_batched_attention as bounded  # noqa: E402


METRICS = {"max_abs_error_by_dtype": {}}
_OLD_THREAD_COUNT = None


def setUpModule():
    global _OLD_THREAD_COUNT
    _OLD_THREAD_COUNT = torch.get_num_threads()
    # Small independent calls are much slower with large CPU thread pools.
    torch.set_num_threads(min(2, _OLD_THREAD_COUNT))


def tearDownModule():
    torch.set_num_threads(_OLD_THREAD_COUNT)
    print("\nSynthetic CPU attention validation: " + json.dumps(METRICS, sort_keys=True))


def old_make_encoder_masks(length, L, max_len, frame_valid=None):
    """Frozen upstream mask oracle, including the original padding fallback."""
    B = length.size(0)
    valid_length = torch.arange(L, device=length.device).expand(B, L) < length.unsqueeze(1)
    pmask = ~valid_length
    if frame_valid is not None:
        assert frame_valid.shape == (B, L)
        pmask = pmask | ~frame_valid
    keep = ~pmask
    if L > max_len:
        half = max_len // 2
        idx = torch.arange(L, device=length.device)
        lo = (idx - half).clamp_min(0).clamp_max(L - max_len)
        hi = (idx + half).clamp_max(L).clamp_min(max_len)
        hi = torch.maximum(hi, lo + 1)
        attn_mask = ~((idx[None, :] >= lo[:, None]) & (idx[None, :] < hi[:, None]))
        cumkeep = keep.cumsum(1, dtype=torch.int32)
        cumkeep = torch.cat([cumkeep.new_zeros(B, 1), cumkeep], dim=1)
        window_has_key = (cumkeep[:, hi] - cumkeep[:, lo]) > 0
    else:
        attn_mask = None
        window_has_key = keep.any(1, keepdim=True)
    return attn_mask, pmask & window_has_key, keep


def old_sdpa_mask(length, L, max_len, frame_valid=None):
    attn_mask, padding, keep = old_make_encoder_masks(length, L, max_len, frame_valid)
    keep_mask = ~padding[:, None, None, :]
    if attn_mask is not None:
        keep_mask = keep_mask & ~attn_mask[None, None, :, :]
    return keep_mask, keep


def dense_oracle(q, k, v, length, max_len, frame_valid=None):
    mask, keep = old_sdpa_mask(length, q.shape[-2], max_len, frame_valid)
    # A fixed reference backend avoids depending on automatic kernel selection.
    # The implementation under test retains PyTorch's automatic selection.
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    return out, keep


def absolute_rope_oracle(x):
    """Absolute frame positions, independently of RoPE's cache or query batches."""
    L, D = x.shape[-2:]
    position = torch.arange(L, dtype=torch.float32, device=x.device)
    frequency = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    angles = torch.einsum("i,j->ij", position, frequency).repeat_interleave(2, dim=-1)
    pairs = x.reshape(*x.shape[:-1], D // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


def dense_rope_attention_oracle(attention, x, mask, query_indices=None):
    B, L, _ = x.shape
    q, k, v = attention.qkv(x).chunk(3, dim=-1)
    q, k, v = [part.reshape(B, L, attention.num_heads, -1).transpose(1, 2)
               for part in (q, k, v)]
    q, k = absolute_rope_oracle(q), absolute_rope_oracle(k)
    if query_indices is not None:
        q = q[:, :, query_indices]
        if mask.shape[-2] != 1:
            mask = mask[:, :, query_indices]
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    out = out.transpose(1, 2).reshape(B, q.shape[-2], -1)
    return attention.proj(out)


def dense_encoder_oracle(encoder, x, length, frame_valid=None):
    mask, keep = old_sdpa_mask(length, x.shape[1], encoder.max_len, frame_valid)
    for block in encoder.blocks:
        x = x + block.gate_msa * dense_rope_attention_oracle(block.attn, block.norm1(x), mask)
        x = x + block.gate_mlp * block.mlp(block.norm2(x))
    return x, keep


def seeded_encoder(*, layers_count=3, width=32, heads=4, max_len=120,
                   chunk_size=64, dropout=0.0):
    with torch.random.fork_rng():
        torch.manual_seed(1409)
        encoder = layers.RoPEEncoder(width, layers_count, heads, 2.0, dropout, max_len,
                                     query_chunk_size=chunk_size)
        with torch.no_grad():
            for block in encoder.blocks:
                # Zero gates would make even completely wrong attention look equal.
                block.gate_msa.uniform_(0.15, 0.65)
                block.gate_mlp.uniform_(-0.3, 0.3)
    return encoder


def max_error(actual, expected):
    if not actual.numel():
        return 0.0
    return (actual.double() - expected.double()).abs().max().item()


class QueryBatchedAttentionTests(unittest.TestCase):
    def assert_numerically_equal(self, actual, expected, dtype):
        tolerance = {
            torch.float64: (2e-12, 2e-12),
            torch.float32: (3e-6, 2e-5),
            torch.float16: (2e-3, 2e-3),
            torch.bfloat16: (2e-2, 2e-2),
        }
        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        atol, rtol = tolerance[dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        label = str(dtype).removeprefix("torch.")
        errors = METRICS["max_abs_error_by_dtype"]
        errors[label] = max(errors.get(label, 0.0), max_error(actual, expected))

    def test_dense_equivalence_multiple_precisions_and_window_boundaries(self):
        # Includes the global/local boundary, an incomplete final query batch,
        # odd windows, width-one self-attention and genuinely empty sequences.
        cases = [
            (0, 120, 8), (1, 1, 1), (17, 120, 5), (120, 120, 32),
            (121, 120, 32), (257, 120, 256), (269, 121, 64),
            (41, 7, 13), (33, 1, 7),
        ]
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            for L, window, chunk in cases:
                for masked_frames in (False, True):
                    with self.subTest(dtype=dtype, L=L, window=window, chunk=chunk,
                                      masked_frames=masked_frames):
                        generator = torch.Generator().manual_seed(913 + L + window)
                        q, k, v = [torch.randn(3, 2, L, 8, generator=generator).to(dtype)
                                   for _ in range(3)]
                        length = torch.tensor([L, max(0, L - max(1, L // 4)), 0])
                        frame_valid = None
                        if masked_frames:
                            frame_valid = torch.ones(3, L, dtype=torch.bool)
                            frame_valid[0, L // 4:3 * L // 4] = False
                            frame_valid[1, ::3] = False
                            frame_valid[2] = False
                        with torch.inference_mode():
                            expected, expected_keep = dense_oracle(q, k, v, length, window,
                                                                   frame_valid)
                            plan, keep = bounded.make_encoder_attention_plan(
                                length, L, window, frame_valid, chunk)
                            actual = bounded.query_batched_attention(q, k, v, plan)
                        torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)
                        self.assert_numerically_equal(actual, expected, dtype)

    def test_padding_fallback_and_every_chunk_have_exact_original_visibility(self):
        for L, window, chunk in [(47, w, c) for w in (1, 3, 7, 48)
                                for c in (1, 8, 31, 64)] + [(121, 120, 32)]:
            with self.subTest(L=L, window=window, chunk=chunk):
                length = torch.tensor([L, L // 2, 0])
                valid = torch.zeros(3, L, dtype=torch.bool)
                valid[0, 0] = True
                valid[0, -1] = True
                valid[1, L // 4] = True
                expected, expected_keep = old_sdpa_mask(length, L, window, valid)
                expected = expected.expand(3, 1, L, L)
                plan, keep = bounded.make_encoder_attention_plan(length, L, window, valid, chunk)
                reconstructed = torch.zeros_like(expected)
                visited = []
                for start, stop, key_start, key_stop in plan.chunks:
                    self.assertLessEqual(stop - start, chunk)
                    self.assertTrue(0 <= key_start < key_stop <= L)
                    local_mask = ~plan.key_padding_mask[:, None, None, key_start:key_stop]
                    if plan.lo is not None:
                        keys = torch.arange(key_start, key_stop)
                        visible = ((keys[None] >= plan.lo[start:stop, None]) &
                                   (keys[None] < plan.hi[start:stop, None]))
                        local_mask = local_mask & visible[None, None]
                    reconstructed[:, :, start:stop, key_start:key_stop] = local_mask
                    visited.extend(range(start, stop))
                self.assertEqual(visited, list(range(L)))
                torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
                torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)

    def test_three_layer_encoder_with_nonzero_gates_matches_dense_oracle(self):
        encoder = seeded_encoder().eval()
        generator = torch.Generator().manual_seed(1713)
        x = torch.randn(3, 385, 32, generator=generator)
        length = torch.tensor([385, 211, 0])
        valid = torch.ones(3, 385, dtype=torch.bool)
        valid[0, 110:280] = False
        valid[1, 30:75] = False
        with torch.inference_mode():
            expected, expected_keep = dense_encoder_oracle(encoder, x, length, valid)
            actual, keep = encoder(x, length, valid)
        self.assertGreater(max_error(expected, x), 0.01)
        torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)
        self.assert_numerically_equal(actual, expected, torch.float32)
        METRICS["three_layer_encoder_max_abs_error"] = max_error(actual, expected)

    def test_absolute_rope_beyond_4096_against_full_key_dense_rows(self):
        L = 4103
        encoder = seeded_encoder(layers_count=1, width=16, heads=2, chunk_size=256).eval()
        attention = encoder.blocks[0].attn
        generator = torch.Generator().manual_seed(629)
        x = torch.randn(1, L, 16, generator=generator)
        length = torch.tensor([L])
        valid = torch.ones(1, L, dtype=torch.bool)
        valid[:, 512:900] = False
        selected = torch.tensor([0, 59, 60, 255, 256, 512, 1024, 4095, 4096, 4097, L - 1])
        with torch.inference_mode():
            plan, _ = bounded.make_encoder_attention_plan(length, L, 120, valid, 256)
            actual = attention(x, attention_plan=plan)[:, selected]
            # Keep all keys in the independent reference, but only materialize
            # score rows at frame/window/cache boundaries.
            full_mask, _ = old_sdpa_mask(length, L, 120, valid)
            expected = dense_rope_attention_oracle(attention, x, full_mask, selected)
        self.assertGreaterEqual(attention.rope._cos_sin_cache[0].shape[0], L)
        self.assert_numerically_equal(actual, expected, torch.float32)
        METRICS["rope_over_4096_max_abs_error"] = max_error(actual, expected)

    def test_10800_frames_have_bounded_calls_and_keep_complete_sequence(self):
        L, chunk, window, depth = 10800, 256, 120, 2
        encoder = seeded_encoder(layers_count=depth, chunk_size=chunk).eval()
        generator = torch.Generator().manual_seed(606)
        x = torch.randn(1, L, 32, generator=generator)
        length = torch.tensor([L])
        valid = torch.ones(1, L, dtype=torch.bool)
        valid[:, 500:900] = False
        real_sdpa = F.scaled_dot_product_attention
        calls = []

        def recording_sdpa(q, k, v, *args, **kwargs):
            mask = kwargs["attn_mask"]
            self.assertLessEqual(q.shape[-2], chunk)
            self.assertLessEqual(k.shape[-2], chunk + window - 1)
            self.assertEqual(mask.shape[-2:], (q.shape[-2], k.shape[-2]))
            calls.append((q.shape[-2], k.shape[-2], mask.numel()))
            return real_sdpa(q, k, v, *args, **kwargs)

        with torch.inference_mode(), \
                mock.patch.object(layers, "make_encoder_masks",
                                  side_effect=AssertionError("long inference allocated dense masks")), \
                mock.patch.object(bounded.F, "scaled_dot_product_attention", side_effect=recording_sdpa):
            actual, keep = encoder(x, length, valid)
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(keep, valid, rtol=0, atol=0)
        self.assertEqual(len(calls), ((L + chunk - 1) // chunk) * depth)
        self.assertEqual(max(q for q, _, _ in calls), 256)
        self.assertEqual(max(k for _, k, _ in calls), 375)
        self.assertEqual(sum(q for q, _, _ in calls), L * depth)
        METRICS["10800_frame_structure"] = {
            "synthetic_encoder_width": 32, "layers": depth,
            "sdpa_calls": len(calls), "max_queries": max(q for q, _, _ in calls),
            "max_keys": max(k for _, k, _ in calls),
            "max_bool_mask_elements": max(n for _, _, n in calls),
            "output_frames": actual.shape[1],
        }

    def test_short_sequences_training_and_enabled_gradients_use_original_path(self):
        cases = [
            # mode, gradient tracking, length, chunk size
            (False, False, 1, 256), (False, False, 256, 256),
            (True, True, 37, 8), (True, False, 37, 8),
            (False, True, 37, 8), (False, False, 37, None),
        ]
        for training, gradients, L, chunk in cases:
            with self.subTest(training=training, gradients=gradients, L=L, chunk=chunk):
                encoder = seeded_encoder(layers_count=1, width=16, heads=2,
                                         max_len=7, chunk_size=chunk, dropout=0.1)
                encoder.train(training)
                x = torch.randn(1, L, 16, requires_grad=gradients)
                length = torch.tensor([L])
                context = torch.enable_grad() if gradients else torch.no_grad()
                with context, \
                        mock.patch.object(layers, "make_encoder_attention_plan",
                                          side_effect=AssertionError("unexpected inference plan")), \
                        mock.patch.object(layers, "make_encoder_masks",
                                          wraps=layers.make_encoder_masks) as dense_masks:
                    output, _ = encoder(x, length)
                    self.assertTrue(torch.isfinite(output).all().item())
                    if gradients:
                        output.square().mean().backward()
                        self.assertTrue(torch.isfinite(x.grad).all().item())
                        self.assertTrue(torch.isfinite(encoder.blocks[0].attn.qkv.weight.grad).all().item())
                dense_masks.assert_called_once()

    def test_inference_helper_rejects_accidental_gradient_use(self):
        plan, _ = bounded.make_encoder_attention_plan(torch.tensor([3]), 3, 1, chunk_size=2)
        with torch.enable_grad():
            q = torch.randn(1, 1, 3, 4, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                bounded.query_batched_attention(q, q, q, plan)

    def test_checkpoint_keys_are_unchanged_by_chunk_setting(self):
        dense = seeded_encoder(chunk_size=None)
        batched = seeded_encoder(chunk_size=256)
        result = batched.load_state_dict(dense.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(list(dense.state_dict()), list(batched.state_dict()))


if __name__ == "__main__":
    unittest.main()

"""10,800-frame structural smoke test using untrained default-size networks.

This exercises the body/foot/hand temporal networks, not image extraction,
licensed SMPL-X decoding, real motion quality, GPU memory or MMD export.
"""

import gc
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ExMR"))
from exmr.model.body_net import BETAS, BodyNet
from exmr.model.refiners import FootRefiner, HandRefiner
from exmr.model import query_batched_attention as attention


class DefaultNetworkLongSequenceTests(unittest.TestCase):
    def test_default_networks_keep_all_10800_frames(self):
        torch.manual_seed(20261104)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        n = 10800
        length = torch.tensor([n])
        cliffcam = torch.randn(1, n, 3)
        hand_features = torch.randn(1, n, 2, 1024)
        metrics = []
        original_sdpa = attention.F.scaled_dot_product_attention
        calls = []

        def bounded_sdpa(q, k, v, **kwargs):
            self.assertLessEqual(q.shape[-2], 256)
            self.assertLessEqual(k.shape[-2], 375)
            calls.append((q.shape[-2], k.shape[-2]))
            return original_sdpa(q, k, v, **kwargs)

        try:
            for factory, args in (
                (BodyNet, lambda: (length, torch.randn(1, n, 23, 3), cliffcam,
                                   torch.randn(1, n, 6), torch.randn(1, n, 1024), hand_features)),
                (FootRefiner, lambda: (length, torch.randn(1, n, 8, 3), cliffcam,
                                       torch.randn(1, n, 24))),
                (HandRefiner, lambda: (length, hand_features, cliffcam,
                                       torch.randn(1, n, 204))),
            ):
                with self.subTest(network=factory.__name__):
                    model = factory().eval()
                    with torch.no_grad():
                        for block in model.encoder.blocks:
                            block.gate_msa.fill_(0.1)
                            block.gate_mlp.fill_(0.1)
                    calls.clear()
                    start = time.perf_counter()
                    with torch.inference_mode(), patch.object(
                        attention.F, "scaled_dot_product_attention", side_effect=bounded_sdpa
                    ):
                        output = model(*args())
                    elapsed = time.perf_counter() - start
                    values = output.values() if isinstance(output, dict) else (output,)
                    for value in values:
                        self.assertEqual(tuple(value.shape[:2]), (1, n))
                        self.assertTrue(bool(torch.isfinite(value).all()))
                    if isinstance(output, dict):
                        beta = output["x"][..., BETAS]
                        self.assertTrue(torch.equal(beta, beta[:, :1].expand_as(beta)))
                    self.assertEqual(len(calls), 43 * len(model.encoder.blocks))
                    metrics.append({"network": factory.__name__, "frames": n,
                                    "layers": len(model.encoder.blocks),
                                    "queries_max": max(q for q, _ in calls),
                                    "keys_max": max(k for _, k in calls),
                                    "sdpa_calls": len(calls), "seconds": round(elapsed, 3)})
                    del model, output, values, value
                    if "beta" in locals():
                        del beta
                    gc.collect()
        finally:
            torch.set_num_threads(previous_threads)
        print("UNTRAINED_DEFAULT_NETWORK_SMOKE " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    unittest.main()

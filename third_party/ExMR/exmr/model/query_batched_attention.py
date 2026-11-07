"""Bounded inference attention with ExMR's original temporal visibility.

Queries are evaluated in batches after absolute-position RoPE. Only keys that
the original window masks out for every query in a batch are omitted. This is
not independent inference on video segments: each encoder layer still reads
the preceding layer's complete sequence.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderAttentionPlan:
    length: int
    # (query start, query stop, key start, key stop), using Python integers so
    # the attention loop does not synchronize the GPU to find its slice bounds.
    chunks: tuple[tuple[int, int, int, int], ...]
    key_padding_mask: torch.Tensor
    lo: torch.Tensor | None
    hi: torch.Tensor | None


def make_encoder_attention_plan(length, L, max_len, frame_valid=None, chunk_size=256):
    """Return an O(B*L) plan and the original valid-observation mask.

    Keep the exact boundary, odd-window and empty-window behavior of
    layers.make_encoder_masks, including its key-padding fallback. In
    particular, do not replace that fallback with a different diagonal mask.
    """
    for name, value in (("chunk_size", chunk_size), ("max_len", max_len)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    B = length.size(0)
    idx = torch.arange(L, device=length.device)
    pmask = ~(idx[None, :] < length[:, None])
    if frame_valid is not None:
        assert frame_valid.shape == (B, L), f"{tuple(frame_valid.shape)} != {(B, L)}"
        pmask = pmask | ~frame_valid
    keep = ~pmask
    lo = hi = None
    if L > max_len:
        half = max_len // 2
        lo = (idx - half).clamp_min(0).clamp_max(L - max_len)
        hi = (idx + half).clamp_max(L).clamp_min(max_len)
        hi = torch.maximum(hi, lo + 1)
        cumkeep = keep.cumsum(1, dtype=torch.int32)
        cumkeep = torch.cat([cumkeep.new_zeros(B, 1), cumkeep], dim=1)
        window_has_key = (cumkeep[:, hi] - cumkeep[:, lo]) > 0
    else:
        window_has_key = keep.any(1, keepdim=True)

    chunks = []
    for start in range(0, L, chunk_size):
        stop = min(start + chunk_size, L)
        if lo is None:
            key_start, key_stop = 0, L
        else:
            # The original lo/hi are monotone, so these endpoints contain
            # every visible key for this entire query batch (including edges).
            key_start = min(max(start - half, 0), L - max_len)
            last_lo = min(max(stop - 1 - half, 0), L - max_len)
            key_stop = max(min(stop - 1 + half, L), max_len, last_lo + 1)
        chunks.append((start, stop, key_start, key_stop))
    plan = EncoderAttentionPlan(L, tuple(chunks), pmask & window_has_key, lo, hi)
    return plan, keep


def query_batched_attention(q, k, v, plan):
    """SDPA on (B,H,L,D), without allocating an L-by-L attention matrix.

    Input precision is unchanged and PyTorch still selects the SDPA backend;
    different tensor shapes can select different kernels and roundoff. The
    caller must disable gradients; training retains the upstream path.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Query-batched ExMR attention is inference-only")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have matching (B,H,L,D) shapes")
    if q.shape[-2] != plan.length or plan.key_padding_mask.shape != (q.shape[0], plan.length):
        raise ValueError("Attention plan does not match q, k and v")
    out = torch.empty_like(q)
    for start, stop, key_start, key_stop in plan.chunks:
        keep_mask = ~plan.key_padding_mask[:, None, None, key_start:key_stop]
        if plan.lo is not None:
            keys = torch.arange(key_start, key_stop, device=q.device)
            visible = ((keys[None, :] >= plan.lo[start:stop, None]) &
                       (keys[None, :] < plan.hi[start:stop, None]))
            keep_mask = keep_mask & visible[None, None, :, :]
        out[:, :, start:stop] = F.scaled_dot_product_attention(
            q[:, :, start:stop], k[:, :, key_start:key_stop], v[:, :, key_start:key_stop],
            attn_mask=keep_mask, dropout_p=0.0,
        )
    return out

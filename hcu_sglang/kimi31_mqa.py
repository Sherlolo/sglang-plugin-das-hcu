import logging
import os
from contextvars import ContextVar
from functools import cache

import torch


_logged = False
_logged_prefill = False
_verify_paged_topk = ContextVar("kimi31_verify_paged_topk", default=False)
_logged_topk_layouts = set()


@cache
def _verify_lightop_device(device):
    props = torch.cuda.get_device_properties(device)
    return (
        getattr(props, "gcnArchName", "").split(":")[0] == "gfx938"
        and props.multi_processor_count == 64
    )


def around_verify_paged_topk(original, self, forward_batch, layer_id, q_fp8, weights, metadata):
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import effective_forward_mode

    selected = (
        effective_forward_mode(forward_batch).is_target_verify()
        and self.num_init_tokens == 0
        and self.num_local_tokens == 0
        and torch.version.hip is not None
        and q_fp8.is_cuda
        and _verify_lightop_device(q_fp8.device)
    )
    token = _verify_paged_topk.set(selected)
    try:
        return original(self, forward_batch, layer_id, q_fp8, weights, metadata)
    finally:
        _verify_paged_topk.reset(token)


def around_verify_raw_topk(original, self, score, lengths, topk, row_starts=None):
    if not _verify_paged_topk.get() or not self.is_torch() or row_starts is not None:
        return original(self, score, lengths, topk, row_starts=row_starts)
    rows, width = score.shape
    max_rows = 128 if os.environ.get("HCU_KIMI31_VERIFY_LIGHTOP_TOPK_LARGE") == "1" else 64
    selected = (
        topk == 2048
        and score.dtype == torch.float32
        and score.is_contiguous()
        and lengths.numel() == rows
        and 8 <= rows <= max_rows
        and 131072 <= width <= 135168
    )
    layout = (rows, width, score.stride(), selected)
    if layout not in _logged_topk_layouts:
        _logged_topk_layouts.add(layout)
        if selected:
            logging.getLogger(__name__).info(
                "DSA Verify local Top-k=LightOp (rows=%d, width=%d)", rows, width
            )
        else:
            logging.getLogger(__name__).info(
                "DSA Verify local Top-k keeps torch (rows=%d, width=%d, stride=%s, k=%d, lengths=%d)",
                rows, width, score.stride(), topk, lengths.numel(),
            )
    if not selected:
        return original(self, score, lengths, topk, row_starts=row_starts)
    from hcu_sglang.layers.attention.current_dsa.dsa_topk_backend import lightop_ranked_local_topk

    return lightop_ranked_local_topk(
        score, lengths, topk,
        triton_rank=os.environ.get("HCU_KIMI31_VERIFY_LIGHTOP_TRITON_RANK") == "1",
    )


def _supports_ragged_mqa(q, kv, weights, ks, ke):
    if torch.version.hip is None or not q.is_cuda or len(kv) != 2:
        return False
    k, scales = kv
    return (
        q.ndim == 3 and k.ndim == 2 and q.shape[-1] == k.shape[-1] == 128
        and q.shape[1] in (8, 16, 32, 64)
        and q.dtype == k.dtype == torch.float8_e4m3fn
        and weights.shape == q.shape[:2] and weights.dtype == torch.float32
        and scales.numel() == k.shape[0] and scales.dtype == torch.float32
        and ks.shape == ke.shape == (q.shape[0],)
        and ks.dtype in (torch.int32, torch.int64)
        and ke.dtype in (torch.int32, torch.int64)
        and all(t.device == q.device for t in (k, scales, weights, ks, ke))
    )


def _pack_ragged_k(k, scales):
    tokens, dim = k.shape
    pages = (tokens + 63) // 64
    padding = pages * 64 - tokens
    payload = torch.nn.functional.pad(k.contiguous().view(torch.uint8), (0, 0, 0, padding))
    scale_bytes = torch.nn.functional.pad(scales.reshape(-1).float(), (0, padding))
    packed = torch.empty((pages, 64 * 132), dtype=torch.uint8, device=k.device)
    packed[:, :64 * dim] = payload.reshape(pages, 64 * dim)
    packed[:, 64 * dim:] = scale_bytes.reshape(pages, 64).contiguous().view(torch.uint8)
    return packed.view(pages, 64, 1, 132)


def _ragged_mqa_logits(q, kv, weights, ks, ke):
    rows, heads, dim = q.shape
    width = kv[0].shape[0]
    if rows == 0 or width == 0:
        return torch.empty((rows, width), device=q.device, dtype=torch.float32)
    from deepgemm import paged_mqa_logits

    packed = _pack_ragged_k(*kv)
    table = torch.arange(packed.shape[0], device=q.device, dtype=torch.int32)
    table = table.unsqueeze(0).expand(rows, -1)
    logits = paged_mqa_logits(
        q.contiguous().view(rows, 1, heads, dim), packed, weights.contiguous(),
        ke.to(torch.int32).contiguous(), table, width,
        clean_logits=True, is_prefill=True,
    )
    columns = torch.arange(width, device=q.device)
    logits.masked_fill_(columns.unsqueeze(0) < ks.unsqueeze(1), float('-inf'))
    return logits


def around_ragged_mqa(original, self, q, kv, weights, ks, ke):
    if not _supports_ragged_mqa(q, kv, weights, ks, ke):
        return original(self, q, kv, weights, ks, ke)
    global _logged_prefill
    if not _logged_prefill:
        logging.getLogger(__name__).info(
            'Kimi31 DeepGEMM prefill MQA: q=%s k=%s weighted per-head ReLU, ragged bounds',
            tuple(q.shape), tuple(kv[0].shape))
        _logged_prefill = True
    with torch.profiler.record_function('kimi31::deepgemm_prefill_mqa'):
        return _ragged_mqa_logits(q, kv, weights, ks, ke)


def around_paged_mqa(original, self, q, cache, weights, context_lens,
                     block_table, schedule_metadata, max_len, clean_logits=False):
    if (q.dtype != torch.float8_e4m3fn or cache.dtype != torch.uint8
            or cache.shape[-1] != 132 or cache.shape[1] != 64
            or q.shape[-1] != 128 or q.shape[-2] not in (8, 16, 32, 64)):
        return original(self, q, cache, weights, context_lens, block_table,
                        schedule_metadata, max_len, clean_logits=clean_logits)
    from deepgemm import paged_mqa_logits

    batch, rows, heads, dim = q.shape
    if batch == 0:
        return torch.empty((0, max_len), device=q.device, dtype=torch.float32)
    if context_lens.numel() == batch * rows:
        lens = context_lens.reshape(-1)
    elif context_lens.numel() == batch:
        lens = (context_lens.reshape(batch, 1) - rows + 1
                + torch.arange(rows, device=q.device, dtype=context_lens.dtype)).reshape(-1)
    else:
        raise ValueError('MQA context lengths do not match query rows')
    if rows > 1:
        block_table = block_table.repeat_interleave(rows, dim=0)
    global _logged
    if not _logged:
        logging.getLogger(__name__).info(
            'Kimi31 DeepGEMM paged_mqa_logits: query=%s cache=%s independent causal rows',
            tuple(q.shape), tuple(cache.shape))
        _logged = True
    return paged_mqa_logits(
        q.reshape(batch * rows, 1, heads, dim).contiguous(), cache,
        weights.reshape(batch * rows, heads).float().contiguous(),
        lens.to(torch.int32).contiguous(), block_table.to(torch.int32), max_len,
        clean_logits=True,
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if os.environ.get("HCU_KIMI31_VERIFY_LIGHTOP_TOPK") == "1":
        HookRegistry.register(
            "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged",
            around_verify_paged_topk, HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.layers.attention.dsa.dsa_topk_backend.DSATopKBackend.topk_func",
            around_verify_raw_topk, HookType.AROUND,
        )
    if os.environ.get('HCU_KIMI31_DEEPGEMM_MQA') == '1':
        HookRegistry.register(
            'sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._paged_mqa_logits',
            around_paged_mqa, HookType.AROUND,
        )
    if os.environ.get('HCU_KIMI31_DEEPGEMM_PREFILL_MQA') == '1':
        HookRegistry.register(
            'sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._mqa_logits',
            around_ragged_mqa, HookType.AROUND,
        )

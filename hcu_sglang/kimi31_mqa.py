import logging
import os

import torch


_logged = False


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
    if os.environ.get('HCU_KIMI31_DEEPGEMM_MQA') != '1':
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        'sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._paged_mqa_logits',
        around_paged_mqa, HookType.AROUND,
    )

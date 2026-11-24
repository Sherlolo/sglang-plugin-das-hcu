import logging
import os

import torch


_logged_heads = set()


def around_kimi_backends(original, view):
    if (torch.version.hip is None
            or view.dsa_prefill_backend != 'flashmla_sparse'
            or view.dsa_decode_backend != 'flashmla_kv'):
        return original(view)
    if view.enable_hisparse:
        raise ValueError('Kimi31 sparse prefill does not support HiSparse')
    from sglang.srt.arg_groups.model_override_base import ResolvedView

    validated = ResolvedView(view, overlay={'dsa_prefill_backend': 'flashmla_kv'})
    declared = original(validated)
    return dict(declared, dsa_prefill_backend='flashmla_sparse')


def _forward_padded_sparse(original, self, q_all, kv_cache, v_head_dim,
                           page_table_1, sm_scale, topk_length=None,
                           attn_sink=None, allow_small_heads=False):
    heads = q_all.shape[1]
    padded_heads = ((heads + 63) // 64) * 64
    q = q_all.new_zeros((q_all.shape[0], padded_heads, q_all.shape[2]))
    q[:, :heads] = q_all
    sink = attn_sink
    if sink is not None:
        sink = attn_sink.new_zeros(padded_heads)
        sink[:heads] = attn_sink
    if heads not in _logged_heads:
        logging.getLogger(__name__).info(
            'Kimi31 FlashMLA sparse prefill pads heads %d to %d', heads, padded_heads)
        _logged_heads.add(heads)
    with torch.profiler.record_function('kimi31::flashmla_sparse_prefill'):
        output = original(
            self, q, kv_cache, v_head_dim, page_table_1, sm_scale,
            topk_length=topk_length, attn_sink=sink,
            allow_small_heads=allow_small_heads,
        )
    return output[:, :heads]


def around_sparse_prefill(original, self, q_all, kv_cache, v_head_dim,
                          page_table_1, sm_scale, topk_length=None,
                          attn_sink=None, allow_small_heads=False):
    heads = q_all.shape[1] if q_all.ndim == 3 else 0
    supported = (
        torch.version.hip is not None and q_all.is_cuda
        and q_all.ndim == 3 and q_all.dtype == torch.bfloat16
        and kv_cache.dtype == torch.bfloat16 and v_head_dim == 512
        and q_all.shape[-1] in (512, 576) and 0 < heads <= 128
    )
    needs_padding = supported and heads % 64 != 0 and 64 % heads != 0
    if needs_padding:
        return _forward_padded_sparse(
            original, self, q_all, kv_cache, v_head_dim, page_table_1,
            sm_scale, topk_length, attn_sink, allow_small_heads)
    return original(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale,
                    topk_length=topk_length, attn_sink=attn_sink,
                    allow_small_heads=allow_small_heads)


def register_hooks():
    if os.environ.get('HCU_KIMI31_FLASHMLA_SPARSE_HEADS') != '1':
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        'sglang.srt.arg_groups.overrides._kimi_k31_dsa_backends',
        around_kimi_backends, HookType.AROUND,
    )
    HookRegistry.register(
        'sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_flashmla_sparse',
        around_sparse_prefill, HookType.AROUND,
    )

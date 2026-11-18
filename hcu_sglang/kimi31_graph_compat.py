# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import torch


class SparseFlashMLAMetadata:
    def __init__(self, scheduler, geometry):
        if not hasattr(scheduler, 'have_initialized'):
            raise TypeError('HCU FlashMLA requires a scheduler object')
        self.flashmla_metadata = scheduler
        self.num_splits = None
        self.geometry = geometry
        scheduler._kimi31_static_sparse_graph = True

    def slice(self, sli):
        if not isinstance(sli, slice) or sli.start not in (None, 0) or sli.step not in (None, 1):
            raise ValueError('HCU sparse graph metadata supports prefix slices only')
        return self

    def copy_(self, other):
        if not isinstance(other, SparseFlashMLAMetadata):
            raise TypeError('HCU sparse graph metadata source must use the same contract')
        if self.geometry != other.geometry:
            raise ValueError('HCU sparse graph scheduler geometry changed within a bucket')
        if not self.flashmla_metadata.have_initialized:
            self.flashmla_metadata = other.flashmla_metadata


def around_compute_flashmla_metadata(original, self, cache_seqlens, seq_len_q, num_heads_q=None):
    metadata = original(self, cache_seqlens, seq_len_q, num_heads_q=num_heads_q)
    scheduler = metadata.flashmla_metadata
    if isinstance(scheduler, torch.Tensor):
        return metadata
    heads = self.flashmla_kv_num_q_heads if num_heads_q is None else num_heads_q
    geometry = (tuple(cache_seqlens.shape), seq_len_q, heads, self.dsa_index_topk)
    return SparseFlashMLAMetadata(scheduler, geometry)


def validate_sparse_scheduler(kwargs):
    scheduler = kwargs.get('tile_scheduler_metadata')
    if not getattr(scheduler, '_kimi31_static_sparse_graph', False):
        return
    if (kwargs.get('indices') is None or not kwargs.get('is_fp8_kvcache')
            or kwargs.get('topk_length') is not None
            or kwargs.get('extra_topk_length') is not None
            or kwargs.get('extra_k_cache') is not None
            or kwargs.get('num_splits') is not None):
        raise ValueError('Kimi NCCL graph scheduler requires fixed-shape FP8 sparse FlashMLA without length overrides')


def around_kda_packed_decode(original, self, mixed_qkv, a, b, **kwargs):
    if mixed_qkv.shape[0] == 0:
        return mixed_qkv.new_empty((1, 0, kwargs['num_v_heads'], kwargs['head_v_dim']))
    return original(self, mixed_qkv, a, b, **kwargs)


def register_hooks():
    transport = os.environ.get('HCU_KIMI31_EP_TRANSPORT')
    enabled = (transport == 'nccl' and os.environ.get('HCU_KIMI31_NCCL_EP_FULL_GRAPH') == '1') or (
        transport == 'deepep' and os.environ.get('HCU_KIMI31_FULL_GRAPH_COMPAT') == '1')
    if not enabled:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        'sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._compute_flashmla_metadata',
        around_compute_flashmla_metadata, HookType.AROUND,
    )
    HookRegistry.register(
        'sglang.srt.layers.attention.linear.kernels.kda_triton.TritonKDAKernel.packed_decode',
        around_kda_packed_decode, HookType.AROUND,
    )

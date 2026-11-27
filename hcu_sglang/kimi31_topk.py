import logging
import os


_logged = set()


def around_transform(original, self, logits, lengths, topk,
                     topk_transform_method, attn_metadata,
                     cu_seqlens_q_topk=None, topk_indices_offset=None,
                     row_starts=None, batch_idx_list=None,
                     force_unfused_topk=False, packed_extend_topk=False, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not (is_hcu() and self.is_sgl_kernel()
            and os.environ.get("SGLANG_DSA_FUSE_TOPK") == "1"):
        return original(
            self, logits, lengths, topk, topk_transform_method, attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            topk_indices_offset=topk_indices_offset, row_starts=row_starts,
            batch_idx_list=batch_idx_list,
            force_unfused_topk=force_unfused_topk,
            packed_extend_topk=packed_extend_topk, **kwargs,
        )
    if force_unfused_topk:
        return type(self).TORCH.topk_func(logits, lengths, topk, row_starts=row_starts)
    from lightop import fast_topk_transform_fused, fast_topk_transform_ragged_fused
    name = topk_transform_method.name
    if name not in _logged:
        _logged.add(name)
        logging.getLogger(__name__).info("Kimi31 DSA top-k=LightOp fused %s", name)
    if name == "PAGED":
        pages = attn_metadata.page_table_1
        if pages is None:
            raise ValueError("LightOp paged top-k requires page_table_1")
        if batch_idx_list is not None:
            pages = pages[batch_idx_list]
        return fast_topk_transform_fused(
            score=logits, lengths=lengths, page_table_size_1=pages,
            cu_seqlens_q=cu_seqlens_q_topk, topk=topk, row_starts=row_starts,
        )
    if name == "RAGGED":
        if topk_indices_offset is None:
            raise ValueError("LightOp ragged top-k requires topk_indices_offset")
        return fast_topk_transform_ragged_fused(
            score=logits, lengths=lengths, topk_indices_offset=topk_indices_offset,
            topk=topk, row_starts=row_starts,
        )
    raise ValueError(f"Unsupported LightOp top-k transform: {name}")


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_topk_backend.DSATopKBackend.topk_transform",
        around_transform, HookType.AROUND,
    )

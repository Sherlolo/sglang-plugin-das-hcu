import logging

import torch


logger = logging.getLogger(__name__)
_logged_fused = False
_logged_fallback = set()


def _fallback_reason(q, k, v, g, beta, *, ssm_states, cache_indices,
                     query_start_loc, A_log, dt_bias, lower_bound,
                     extend_seq_lens_cpu, beta_is_raw,
                     is_spec_decode, return_intermediate_states):
    if return_intermediate_states or is_spec_decode:
        return "state_snapshot_or_speculation"
    if lower_bound is None or not -5.0 <= float(lower_bound) <= 0.0:
        return "unsupported_gate"
    if not beta_is_raw:
        return "activated_beta"
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, g, beta)):
        return "compute_dtype"
    if (q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128
            or v.shape != q.shape or k.shape != q.shape or g.shape != q.shape
            or beta.shape != q.shape[:-1]):
        return "compute_shape"
    if (ssm_states.ndim != 4
            or ssm_states.dtype not in (torch.bfloat16, torch.float32)
            or ssm_states.shape[1:] != (q.shape[2], 128, 128)
            or cache_indices.ndim != 1 or query_start_loc.ndim != 1
            or cache_indices.dtype not in (torch.int32, torch.int64)
            or query_start_loc.dtype not in (torch.int32, torch.int64)
            or query_start_loc.numel() != cache_indices.numel() + 1
            or A_log is None or A_log.numel() != q.shape[2]
            or dt_bias is None or dt_bias.numel() != q.shape[2] * 128):
        return "state_layout"
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return "graph_capture"
    offsets = query_start_loc.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != q.shape[1]:
        return "sequence_offsets"
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if extend_seq_lens_cpu is not None:
        declared = (extend_seq_lens_cpu.detach().cpu().tolist()
                    if torch.is_tensor(extend_seq_lens_cpu)
                    else list(extend_seq_lens_cpu))
        if declared != lengths:
            return "sequence_offsets"
    if (len(lengths) != cache_indices.numel()
            or any(length < 64 or length > 2048 for length in lengths)):
        return "sequence_lengths"
    slots = cache_indices.detach().cpu().tolist()
    if (len(set(slots)) != len(slots)
            or any(slot < 0 or slot >= ssm_states.shape[0] for slot in slots)):
        return "state_indices"
    return None


def _hcu_extend(self, q, k, v, g, beta, *, ssm_states,
                cache_indices, query_start_loc, **kwargs):
    if q.shape[1] == 0:
        if v.shape[1] != 0:
            raise ValueError("empty FlashKDA batch has nonempty values")
        empty = v.new_empty(v.shape)
        return (empty, None) if kwargs.get("return_intermediate_states") else empty
    if cache_indices.numel() == 0:
        raise ValueError("nonempty FlashKDA batch has no state slots")

    reason = _fallback_reason(
        q, k, v, g, beta, ssm_states=ssm_states,
        cache_indices=cache_indices, query_start_loc=query_start_loc,
        A_log=kwargs.get("A_log"), dt_bias=kwargs.get("dt_bias"),
        lower_bound=kwargs.get("lower_bound"),
        extend_seq_lens_cpu=kwargs.get("extend_seq_lens_cpu"),
        beta_is_raw=kwargs.get("beta_is_raw", False),
        is_spec_decode=kwargs.get("is_spec_decode", False),
        return_intermediate_states=kwargs.get("return_intermediate_states", False),
    )
    if reason is not None:
        if reason not in _logged_fallback:
            logger.info("Kimi31 FlashKDA fallback to Triton KDA: %s", reason)
            _logged_fallback.add(reason)
        from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

        return TritonKDAKernel().extend(
            q, k, v, g, beta, ssm_states=ssm_states,
            cache_indices=cache_indices, query_start_loc=query_start_loc, **kwargs
        )

    global _logged_fused
    if not _logged_fused:
        logger.info("Kimi31 FlashKDA fused HCU prefill selected")
        _logged_fused = True
    return self._flashkda_extend(
        q, k, v, g, beta, ssm_states=ssm_states,
        cache_indices=cache_indices, query_start_loc=query_start_loc,
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"], beta_is_raw=True,
    )


def around_extend(original, self, q, k, v, g, beta, *, ssm_states,
                  cache_indices, query_start_loc, **kwargs):
    if torch.version.hip is None or not q.is_cuda:
        return original(self, q, k, v, g, beta, ssm_states=ssm_states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc, **kwargs)
    return _hcu_extend(self, q, k, v, g, beta, ssm_states=ssm_states,
                       cache_indices=cache_indices,
                       query_start_loc=query_start_loc, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.linear.kernels.kda_flashkda.FlashKDAKernel.extend",
        around_extend,
        HookType.AROUND,
    )

import logging
import os
from contextvars import ContextVar

import torch


logger = logging.getLogger(__name__)
_logged_fused = False
_logged_fallback = set()
_logged_routes = set()
_logged_metadata_cache = False
_forward_metadata_cache = ContextVar('kimi31_forward_metadata_cache', default=None)


def _metadata_list(tensor):
    cache = _forward_metadata_cache.get()
    if cache is None:
        return tensor.detach().cpu().tolist()
    version = None if torch.is_inference(tensor) else tensor._version
    key = (id(tensor), version, tuple(tensor.shape), tuple(tensor.stride()),
           tensor.dtype, tensor.device)
    entry = cache.get(key)
    if entry is None:
        entry = (tensor, tensor.detach().cpu().tolist())
        cache[key] = entry
    return list(entry[1])


def around_model_forward(original, self, forward_batch, *args, **kwargs):
    mode = forward_batch.forward_mode
    if (torch.version.hip is None
            or getattr(self, 'is_draft_worker', False)
            or not mode.is_extend_without_speculative()):
        return original(self, forward_batch, *args, **kwargs)
    global _logged_metadata_cache
    if not _logged_metadata_cache:
        logger.info('Kimi31 forward-scoped FlashKDA metadata CPU cache selected')
        _logged_metadata_cache = True
    token = _forward_metadata_cache.set({})
    try:
        return original(self, forward_batch, *args, **kwargs)
    finally:
        _forward_metadata_cache.reset(token)


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
    offsets = _metadata_list(query_start_loc)
    if offsets[0] != 0 or offsets[-1] != q.shape[1]:
        return "sequence_offsets"
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if extend_seq_lens_cpu is not None:
        declared = (extend_seq_lens_cpu.detach().cpu().tolist()
                    if torch.is_tensor(extend_seq_lens_cpu)
                    else list(extend_seq_lens_cpu))
        if declared != lengths:
            return "sequence_offsets"
    max_length = 131072 if os.environ.get('HCU_KIMI31_FLASHKDA_LONG_PREFILL') == '1' else 2048
    if (len(lengths) != cache_indices.numel()
            or any(length < 64 or length > max_length for length in lengths)):
        return "sequence_lengths"
    slots = _metadata_list(cache_indices)
    if (len(set(slots)) != len(slots)
            or any(slot < 0 or slot >= ssm_states.shape[0] for slot in slots)):
        return "state_indices"
    return None


def _long_extend(q, k, v, g, beta, *, ssm_states, cache_indices,
                 query_start_loc, A_log, dt_bias, lower_bound):
    import flash_kda

    offsets = _metadata_list(query_start_loc)
    lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    batch = len(lengths)
    fixed = len(set(lengths)) == 1
    heads, dim = q.shape[2:]
    compute = [t.contiguous() for t in (q, k, v, g, beta)]
    if fixed:
        compute = [t.view(batch, lengths[0], *t.shape[2:]) for t in compute]
    initial = ssm_states[cache_indices].contiguous()
    final = torch.empty_like(initial)
    output = torch.empty_like(compute[2])
    cp_flag = os.environ.get('HCU_KIMI31_FLASHKDA_FIXED_CP')
    if cp_flag not in (None, '0', '1'):
        raise ValueError('HCU_KIMI31_FLASHKDA_FIXED_CP must be 0 or 1')
    enable_cp = None if cp_flag is None else cp_flag == '1'
    dispatch = {}
    with torch.profiler.record_function('kimi31::flashkda_prefill'):
        flash_kda.fwd(
            *compute, dim ** -0.5, output,
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(heads, dim).float().contiguous(), lower_bound,
            initial_state=initial, final_state=final,
            cu_seqlens=None if fixed else query_start_loc.to(torch.int64),
            enable_cp=enable_cp, _dispatch_info=dispatch,
        )
        ssm_states[cache_indices] = final
    route = (dispatch.get('route', 'unknown'), batch, tuple(q.shape), fixed)
    if route not in _logged_routes:
        logger.info('Kimi31 FlashKDA prefill route=%s batch=%d q=%s fixed=%s reason=%s',
                    *route, dispatch.get('reason', 'unknown'))
        _logged_routes.add(route)
    return output.reshape(v.shape)


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
    if os.environ.get('HCU_KIMI31_FLASHKDA_LONG_PREFILL') == '1':
        return _long_extend(
            q, k, v, g, beta, ssm_states=ssm_states,
            cache_indices=cache_indices, query_start_loc=query_start_loc,
            A_log=kwargs['A_log'], dt_bias=kwargs['dt_bias'],
            lower_bound=kwargs['lower_bound'],
        )
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

    if os.environ.get('HCU_KIMI31_CPU_METADATA_CACHE') == '1':
        HookRegistry.register(
            'sglang.srt.model_executor.model_runner.ModelRunner.forward',
            around_model_forward,
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.layers.attention.linear.kernels.kda_flashkda.FlashKDAKernel.extend",
        around_extend,
        HookType.AROUND,
    )

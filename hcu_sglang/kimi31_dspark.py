import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path


_router_lightop_decode = ContextVar("kimi31_router_lightop_decode", default=False)

def normalize_speculator(values):
    from transformers import Qwen3Config

    backbone = dict(values["transformer_layer_config"])
    config = Qwen3Config(**backbone)
    for key, value in values.items():
        if key not in ("transformer_layer_config", "auto_map", "speculators_config"):
            setattr(config, key, value)
    config.architectures = ["DSparkDraftModel"]
    config.target_layer_ids = [i - 1 for i in values["aux_hidden_state_layer_ids"]]
    if any(i < 0 for i in config.target_layer_ids):
        raise ValueError("DSpark auxiliary capture layer IDs must be positive")
    config.is_causal = not values.get("sliding_window_non_causal", False)
    config.dflash_config = {
        "target_layer_ids": config.target_layer_ids,
        "block_size": values["block_size"],
        "mask_token_id": values["mask_token_id"],
    }
    config.dspark_config = {
        k: values[k] for k in (
            "block_size", "mask_token_id", "markov_rank", "markov_head_type",
            "enable_confidence_head", "confidence_head_with_markov", "sample_from_anchor"
        ) if k in values
    }
    logging.getLogger(__name__).info(
        "Kimi31 DSpark config: target_layer_ids=%s, layers=%s, window=%s, causal=%s, gamma=%s",
        config.target_layer_ids, config.num_hidden_layers, config.sliding_window,
        config.is_causal, config.block_size,
    )
    return config


def around_parse(original, self, model, *args, **kwargs):
    path = Path(model) / "config.json"
    if path.is_file():
        values = json.loads(path.read_text())
        if (values.get("speculators_model_type") == "dspark"
                and "transformer_layer_config" in values):
            return normalize_speculator(values)
    return original(self, model, *args, **kwargs)


@contextmanager
def around_draft_context(original, self):
    from sglang.srt.runtime_context import get_parallel

    with original(self):
        if self._draft_is_moe:
            yield
        else:
            with get_parallel().override(disable_attn_tp_gather=True):
                yield


def register_dp_deepep():
    if (os.environ.get("HCU_KIMI31_DEEPEP") != "1"
            or os.environ.get("HCU_KIMI31_EP_TRANSPORT", "deepep") != "deepep"):
        return

    from sglang.srt.arg_groups import model_override_base

    register = getattr(model_override_base, "register_dspark_dp_a2a_backends", None)
    if register is None:
        return
    architecture = "KimiK31ForConditionalGeneration"
    backends = model_override_base.dspark_dp_a2a_backends(architecture)
    if "deepep" not in backends:
        register(architecture, (*backends, "deepep"))


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    register_dp_deepep()
    HookRegistry.register(
        "sglang.srt.speculative.dflash_info_v2.DFlashDraftInputV2.merge_batch",
        around_draft_merge, HookType.AROUND,
    )
    if os.environ.get("HCU_KIMI31_ROUTER_GRAPH_DUMP_DIR"):
        from hcu_sglang import kimi31_router_graph_diag

        backend = "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend"
        HookRegistry.register(f"{backend}.capture_one", kimi31_router_graph_diag.capture_one, HookType.AROUND)
        HookRegistry.register(f"{backend}.replay", kimi31_router_graph_diag.replay, HookType.AROUND)
    if os.environ.get("HCU_KIMI31_ROUTER_LIGHTOP") == "1":
        HookRegistry.register(
            "sglang.srt.models.kimi_k3.KimiK31ForConditionalGeneration.forward",
            around_lightop_router_mode, HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.models.deepseek_v2.MoEGate.forward",
            around_lightop_router, HookType.AROUND,
        )
    if os.environ.get("HCU_KIMI31_HISPARSE_DSPARK") == "1":
        from hcu_sglang import kimi31_hisparse_dspark

        kimi31_hisparse_dspark.register_hooks()
        HookRegistry.register(
            "sglang.srt.managers.tp_worker.TpModelWorker.get_worker_info",
            around_hisparse_decode_request_limit, HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.mem_cache.allocator.hisparse.HiSparseTokenToKVPoolAllocator.max_new_tokens_for_memory",
            around_hisparse_decode_generation_budget, HookType.AROUND,
        )

    HookRegistry.register(
        "sglang.srt.hf_transformers.config.HfModelConfigParser.parse",
        around_parse, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2._draft_context",
        around_draft_context, HookType.AROUND,
    )


def around_draft_merge(original, draft, incoming):
    from copy import copy
    from sglang.srt.runtime_context import get_schedule

    if get_schedule().disable_overlap_schedule:
        if draft.nxt_kv_lens_cpu is None and incoming.nxt_kv_lens_cpu is not None:
            # A cache covering only one side cannot index every merged request.
            # Decode preparation rebuilds both lengths after filtering the merged batch.
            incoming = copy(incoming)
            incoming.nxt_kv_lens_cpu = None
            incoming.nxt_kv_lens_sum = None
        elif draft.nxt_kv_lens_cpu is not None and incoming.nxt_kv_lens_cpu is None:
            draft.nxt_kv_lens_cpu = None
            draft.nxt_kv_lens_sum = None
    return original(draft, incoming)


def around_hisparse_decode_request_limit(original, worker):
    from sglang.srt.runtime_context import get_disagg

    info = original(worker)
    runner = worker.model_runner
    if not runner.enable_hisparse or get_disagg().disaggregation_mode != "decode":
        return info
    host_backed_capacity = getattr(runner.token_to_kv_pool_allocator, "size_full", None)
    if not host_backed_capacity:
        return info
    request_table_limit = runner.req_to_token_pool.max_context_len
    max_request = min(worker.model_config.context_len, request_table_limit, host_backed_capacity) - 1
    if max_request <= info[4]:
        return info
    reserved = info[4] - info[5]
    logging.getLogger(__name__).info(
        "HiSparse Decode request limit uses full KV pool: %d -> %d; logical rows=%d, request table=%d",
        info[5], max_request - reserved, host_backed_capacity, request_table_limit,
    )
    return (*info[:4], max_request, max_request - reserved, *info[6:])


def around_hisparse_decode_generation_budget(original, allocator, input_tokens, max_new_tokens, **kwargs):
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        kwargs["token_capacity"] = allocator.size_full
    return original(allocator, input_tokens, max_new_tokens, **kwargs)


def around_lightop_router_mode(original, model, input_ids, positions, forward_batch, *args, **kwargs):
    mode = forward_batch.forward_mode
    scope = _router_lightop_decode.set(mode.is_target_verify() or mode.is_decode())
    try:
        return original(model, input_ids, positions, forward_batch, *args, **kwargs)
    finally:
        _router_lightop_decode.reset(scope)


def around_lightop_router(original, gate, hidden_states, *args, **kwargs):
    if not _router_lightop_decode.get():
        return original(gate, hidden_states, *args, **kwargs)

    import torch
    from sglang.srt.runtime_context import get_exec

    weight = gate.weight
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[1] != 7168
        or not 0 < hidden_states.shape[0] <= gate.tiny_router_gemm_max_tokens
        or weight.shape != (896, 7168)
        or hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or not hidden_states.is_contiguous()
        or not weight.is_contiguous()
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return original(gate, hidden_states, *args, **kwargs)
    return _lightop_router_provider(hidden_states.shape[0])(
        hidden_states, weight, out_dtype=torch.float32,
    )


@lru_cache(maxsize=None)
def _lightop_router_provider(rows):
    import lightop

    logging.getLogger(__name__).info(
        "Kimi31 Router uses LightOp for M=%d N=896 K=7168, BF16 to FP32", rows,
    )
    return lightop.gemm

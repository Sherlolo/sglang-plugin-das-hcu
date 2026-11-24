import json
import os
from pathlib import Path

import torch


def around_attention_dcp_size(original, **kwargs):
    if (torch.version.hip is not None and kwargs["backend"] == "triton"
            and kwargs["is_draft_worker"] and kwargs["is_dspark"]):
        kwargs = dict(kwargs, is_hip=True)
    return original(**kwargs)


def around_nvlink_clique(original, *args, **kwargs):
    if torch.version.hip is not None:
        return False
    return original(*args, **kwargs)


def around_model_provider(original, fn, server_args, hf_config):
    from sglang.srt.arg_groups.model_override_base import ResolvedView, resolving_view

    cfg = resolving_view(server_args)
    if (os.environ.get("HCU_KIMI31_DCP_FAKE") == "1"
            and torch.version.hip is not None
            and fn.__module__ == "sglang.srt.arg_groups.model_overrides.kimi_k3"
            and fn.__name__ == "_kimi_k31_overrides"
            and cfg.dcp_size > 1 and cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_transfer_backend == "fake"):
        validated = ResolvedView(server_args, overlay={"disaggregation_transfer_backend": "mooncake"})
        return original(fn, validated, hf_config)
    return original(fn, server_args, hf_config)


def after_alloc_memory_pool(result, runner, *args, **kwargs):
    directory = os.environ.get("HCU_KIMI31_KV_INVENTORY")
    if not directory:
        return result
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tensors = []
    seen_objects = set()
    seen_storage = set()

    def visit(value, name, depth):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key in seen_storage:
                return
            seen_storage.add(key)
            tensors.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                                device=str(value.device), bytes=storage.nbytes()))
        elif depth > 0 and isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f"{name}.{i}", depth - 1)
        elif depth > 0 and isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}", depth - 1)
        elif depth > 0 and hasattr(value, "__dict__") and type(value).__module__.startswith(("sglang.srt.mem_cache", "hcu_sglang.mem_cache")):
            for key, item in vars(value).items():
                visit(item, f"{name}.{key}", depth - 1)

    visit(runner.token_to_kv_pool, "kv_pool", 7)
    rank = torch.distributed.get_rank()
    record = dict(rank=rank, draft=runner.is_draft_worker,
                  dcp_size=parallel.attn_dcp_size, dcp_rank=parallel.attn_dcp_rank,
                  max_total_num_tokens=runner.max_total_num_tokens,
                  logical_max_total_num_tokens=runner.logical_max_total_num_tokens,
                  max_running_requests=runner.max_running_requests,
                  pool_type=type(runner.token_to_kv_pool).__name__,
                  allocator_page_size=runner.token_to_kv_pool_allocator.page_size,
                  tensors=tensors, pool_storage_bytes=sum(t["bytes"] for t in tensors),
                  allocated_bytes=torch.cuda.memory_allocated(),
                  reserved_bytes=torch.cuda.memory_reserved())
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    role = "draft" if runner.is_draft_worker else "target"
    (output / f"{role}-rank{rank}.json").write_text(json.dumps(record, indent=2))
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target in (
        "sglang.srt.configs.attention_execution.attention_dcp_size",
        "sglang.srt.layers.attention.triton_backend.attention_dcp_size",
        "sglang.srt.mem_cache.kv_cache_configurator.attention_dcp_size",
    ):
        HookRegistry.register(target, around_attention_dcp_size, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_dcp_sharded.is_one_nvlink_clique",
        around_nvlink_clique, HookType.AROUND,
    )
    if os.environ.get("HCU_KIMI31_DCP_FAKE") == "1":
        HookRegistry.register(
            "sglang.srt.arg_groups.overrides._invoke_provider",
            around_model_provider, HookType.AROUND,
        )
    if os.environ.get("HCU_KIMI31_KV_INVENTORY"):
        HookRegistry.register(
            "sglang.srt.model_executor.model_runner.ModelRunner.alloc_memory_pool",
            after_alloc_memory_pool, HookType.AFTER,
        )

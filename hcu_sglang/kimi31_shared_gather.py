"""Optional graph-safe intra-Pod gather for Kimi's shared-expert inputs."""
import logging
import os

import torch

_LOG = logging.getLogger(__name__)


def around_moe_init(original, layer, *args, **kwargs):
    original(layer, *args, **kwargs)
    if not layer._shared_experts_tp_comm:
        return
    group = layer._shared_experts_tp_group
    if group.world_size != 4 or hasattr(group, "_kimi31_shared_gather"):
        return

    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    communicator = CustomAllreduce(
        group=group.cpu_group,
        device=group.device,
        enable_register_for_capturing=False,
        max_size=8 * 1024 * 1024,
    )
    group._kimi31_shared_gather = communicator
    _LOG.info("Kimi31 shared expert AITER gather workspace ready disabled=%s", communicator.disabled)


def around_shared_gather(original, layer, hidden_states):
    group = layer._shared_experts_tp_group
    communicator = getattr(group, "_kimi31_shared_gather", None)
    if (
        communicator is None
        or communicator.disabled
        or hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] not in (3584,7168)
        or hidden_states.shape[0] != 32
        or not torch.cuda.is_current_stream_capturing()
        or not communicator.should_custom_ag(hidden_states)
    ):
        return original(layer, hidden_states)

    if not getattr(group, "_kimi31_shared_gather_logged", False):
        _LOG.info("Kimi31 shared expert AITER all-gather active rows=%s width=%s",
                  hidden_states.shape[0], hidden_states.shape[1])
        group._kimi31_shared_gather_logged = True
    output = hidden_states.new_empty((hidden_states.shape[0] * group.world_size, hidden_states.shape[1]))
    communicator.all_gather_unreg(hidden_states, out=output)
    return output


def around_shared_scatter(original, layer, shared_output, hidden_states):
    group = layer._shared_experts_tp_group
    communicator = getattr(group, "_kimi31_shared_gather", None)
    if (
        communicator is None
        or communicator.disabled
        or hidden_states.ndim != 2
        or shared_output.dtype != hidden_states.dtype
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] not in (3584, 7168)
        or hidden_states.shape[0] != 32
        or shared_output.shape != (hidden_states.shape[0] * group.world_size, hidden_states.shape[1])
        or not shared_output.is_contiguous()
        or not hidden_states.is_contiguous()
        or not torch.cuda.is_current_stream_capturing()
        or not communicator.should_custom_ar(shared_output)
    ):
        return original(layer, shared_output, hidden_states)

    if not getattr(group, "_kimi31_shared_scatter_logged", False):
        _LOG.info("Kimi31 shared expert AITER reduce-scatter active rows=%s width=%s",
                  hidden_states.shape[0], hidden_states.shape[1])
        group._kimi31_shared_scatter_logged = True
    output = torch.empty_like(hidden_states)
    communicator.reduce_scatter(shared_output, output, registered=False)
    return output


def around_group_destroy(original, group):
    communicator = getattr(group, "_kimi31_shared_gather", None)
    if communicator is not None:
        group._kimi31_shared_gather = None
        communicator.close()
    return original(group)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    model = "sglang.srt.models.kimi_k3.KimiK3MoE"
    HookRegistry.register(f"{model}.__init__", around_moe_init, HookType.AROUND)
    if os.environ.get("HCU_KIMI31_SHARED_AITER_AG") == "1":
        HookRegistry.register(f"{model}._gather_shared_expert_inputs", around_shared_gather, HookType.AROUND)
    if os.environ.get("HCU_KIMI31_SHARED_AITER_RS") == "1":
        HookRegistry.register(f"{model}._reduce_scatter_shared_experts", around_shared_scatter, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.GroupCoordinator.destroy",
        around_group_destroy, HookType.AROUND,
    )

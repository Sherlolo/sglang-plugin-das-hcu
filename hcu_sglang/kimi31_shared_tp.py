import torch


def around_reduce_scatter(original, self, shared_output, hidden_states):
    group = self._shared_experts_tp_group
    if (group is None or group.world_size != 16
            or shared_output.dtype != torch.bfloat16):
        return original(self, shared_output, hidden_states)
    output = torch.empty_like(hidden_states, dtype=torch.float32)
    group.reduce_scatter_tensor(output, shared_output.float())
    return output.to(hidden_states.dtype)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        'sglang.srt.models.kimi_k3.KimiK3MoE._reduce_scatter_shared_experts',
        around_reduce_scatter, HookType.AROUND,
    )

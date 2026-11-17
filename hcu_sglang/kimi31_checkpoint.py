import logging


def around_get_tensor(original, shard, name):
    from hcu_adapter.slimquant import _ROUTED_EXPERT_RE

    if (shard._active and name.endswith('_scale')
            and _ROUTED_EXPERT_RE.search(name[:-len('_scale')])):
        return shard._f.get_tensor(name)
    return original(shard, name)


def install_checkpoint_loader():
    from hcu_adapter.slimquant import install_weight_loader_shim
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        'hcu_adapter.slimquant._SlimquantShard.get_tensor',
        around_get_tensor, HookType.AROUND,
    )
    install_weight_loader_shim()
    logging.getLogger(__name__).info(
        'Kimi31 checkpoint: dense W8 weights dequantized to BF16 at load; routed INT4 bytes and native scale/16 preserved for DeepGEMM W4A8'
    )

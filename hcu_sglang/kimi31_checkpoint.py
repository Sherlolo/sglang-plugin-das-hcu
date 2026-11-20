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
    import json
    import os
    from functools import lru_cache
    from pathlib import Path
    from hcu_adapter.slimquant import _SlimquantShard

    checkpoint = os.environ.get("HCU_KIMI31_MODELSLIM_CHECKPOINT")
    if checkpoint:
        root = Path(checkpoint).resolve()
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (cfg.get("compression_config") or {}).get("quant_method") != "slimquant_w4a8":
            raise ValueError("The explicitly selected ModelSlim checkpoint must declare slimquant_w4a8")
        original_probe = _SlimquantShard._checkpoint_is_slimquant

        @lru_cache(maxsize=None)
        def is_modelslim_shard(path):
            return bool(path) and Path(path).parent.resolve() == root

        def checkpoint_is_slimquant(cls, path):
            return is_modelslim_shard(path) or original_probe(path)

        _SlimquantShard._checkpoint_is_slimquant = classmethod(checkpoint_is_slimquant)
        logging.getLogger(__name__).info("SlimQuant compression_config support enabled for %s", root)

    install_weight_loader_shim()
    logging.getLogger(__name__).info(
        'Kimi31 checkpoint: dense W8 weights dequantized to BF16 at load; routed INT4 bytes and native scale/16 preserved for DeepGEMM W4A8'
    )

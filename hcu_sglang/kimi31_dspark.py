import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path


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
    if os.environ.get("HCU_KIMI31_HISPARSE_DSPARK") == "1":
        from hcu_sglang import kimi31_hisparse_dspark

        kimi31_hisparse_dspark.register_hooks()

    HookRegistry.register(
        "sglang.srt.hf_transformers.config.HfModelConfigParser.parse",
        around_parse, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2._draft_context",
        around_draft_context, HookType.AROUND,
    )

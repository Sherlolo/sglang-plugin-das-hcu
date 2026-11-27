import logging

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import (
    SlimQuantW4A8Int8MarlinMoEMethod,
)


class Kimi31W4A8DeepEPMoEMethod(FusedMoEMethodBase):
    hcu_kimi31_w4a8_hipc = True
    create_weights = SlimQuantW4A8Int8MarlinMoEMethod.create_weights

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_moe_runner(self, layer, moe_runner_config):
        if moe_runner_config.activation != "situ":
            raise ValueError("Kimi K3.1 W4A8 DeepEP requires SiTU")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError("Kimi K3.1 W4A8 DeepEP requires output router weights")
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def process_weights_after_loading(self, layer):
        from deepgemm import pack_w4a8_moe_hipc_weight
        from hcu_sglang.layers.moe.ct_w4a8 import register_ops

        register_ops()

        self.quant_config.normalize_checkpoint_weights(layer)
        for prefix in ("w13", "w2"):
            name = prefix + "_weight"
            packed = pack_w4a8_moe_hipc_weight(getattr(layer, name).data)
            setattr(layer, name, torch.nn.Parameter(packed, requires_grad=False))
        layer.w4a8_intermediate_size = layer.w13_weight_scale.shape[1] // 2
        layer.w4a8_padded_intermediate_size = layer.w4a8_intermediate_size
        layer.dispatcher.set_quant_config({
            "normal_dispatcher_output_dtype": "int8",
            "low_latency_dispatcher_output_dtype": "int8",
            "normal_expert_alignment": 256,
            "hcu_w4a8_int8_dispatch": True,
            "hcu_w4a8_deepep_ll_producer_event": True,
        })
        logging.getLogger(__name__).info(
            "Kimi31 EP W4A8: DeepGEMM HIPC, INT8 dispatch, SiTU beta=%s linear_beta=%s",
            self.moe_runner_config.gemm1_alpha,
            self.moe_runner_config.gemm1_clamp_limit,
        )

    def apply(self, layer, dispatch_output):
        return layer.run_moe_core(dispatch_output)


def around_get_quant_method(original, config, layer, prefix):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.utils import (
        get_deepep_mode, get_moe_a2a_backend, get_moe_runner_backend,
    )
    from sglang.srt.utils import get_bool_env_var

    if isinstance(layer, LinearBase):
        from hcu_adapter.slimquant_native import native_dense_module
        if native_dense_module(prefix):
            from hcu_sglang.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
            logging.getLogger(__name__).info("Kimi31 native dense INT8: %s", prefix)
            return SlimQuantW4A8Int8LinearMethod(config)
        return UnquantizedLinearMethod()
    if (isinstance(layer, DeepEPMoE) and get_moe_runner_backend().is_deep_gemm()
            and get_moe_a2a_backend().is_deepep()):
        if get_deepep_mode().enable_low_latency() and not get_bool_env_var("SGLANG_USE_W4A8_MASKED_HIPC"):
            raise ValueError("Kimi31 W4A8 low-latency DeepEP requires SGLANG_USE_W4A8_MASKED_HIPC=1")
        if not get_bool_env_var("SGLANG_USE_W4A8_CONTIGUOUS_HIPC"):
            raise ValueError("Kimi31 W4A8 DeepEP requires SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1")
        return Kimi31W4A8DeepEPMoEMethod(config)
    return original(config, layer, prefix)


def around_ep_init(original, layer, *args, **kwargs):
    result = original(layer, *args, **kwargs)
    if getattr(layer.quant_method, "hcu_kimi31_w4a8_hipc", False):
        layer.deprecate_flag = False
    return result


def around_run_moe_core(original, layer, dispatch_output, *args, **kwargs):
    if getattr(layer.quant_method, "hcu_kimi31_w4a8_hipc", False):
        from hcu_sglang.layers.moe.ct_w4a8 import run_moe_core

        return run_moe_core(layer, dispatch_output)
    return original(layer, dispatch_output, *args, **kwargs)


def register_hooks():
    import os
    os.environ["SGLANG_W4A8_TPMOE_BACKEND"] = "aiter"
    from hcu_sglang.layers.moe import deepep_current
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.kimi31_checkpoint import install_checkpoint_loader

    install_checkpoint_loader()

    transport = os.environ.get("HCU_KIMI31_EP_TRANSPORT", "deepep")
    if transport == "nccl":
        from hcu_sglang.kimi31_nccl_ep import Kimi31NcclEPDispatcher
        HookRegistry.register(
            "sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPDispatcher",
            Kimi31NcclEPDispatcher, HookType.REPLACE,
        )
    elif transport == "deepep":
        deepep_current.register_hooks()
    else:
        raise ValueError(f"Unsupported Kimi31 EP transport: {transport}")
    from hcu_sglang.layers.moe.ct_w4a8 import register_ops
    register_ops()
    HookRegistry.register(
        "hcu_sglang.layers.quantization.slimquant_w4a8_marlin.SlimQuantW4A8Int8MarlinConfig.get_quant_method",
        around_get_quant_method, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.__init__",
        around_ep_init, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core",
        around_run_moe_core, HookType.AROUND,
    )


def register_etp_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.kimi31_checkpoint import install_checkpoint_loader

    install_checkpoint_loader()
    HookRegistry.register(
        "hcu_sglang.layers.quantization.slimquant_w4a8_marlin.SlimQuantW4A8Int8MarlinConfig.get_quant_method",
        around_get_quant_method, HookType.AROUND,
    )

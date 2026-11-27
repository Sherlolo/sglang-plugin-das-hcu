import logging
import os
from contextvars import ContextVar


_logged = set()


def status(feature, reason):
    key = (feature, reason)
    if key not in _logged:
        _logged.add(key)
        logging.getLogger(__name__).info("Kimi31 fusion %s: %s", feature, reason)


_mla_front = ContextVar("kimi31_mla_front", default=None)


_pending = ContextVar("kimi31_fusion_pending", default=None)


def enabled(name):
    return os.environ.get(name, "0").lower() in ("1", "true")


def supported(x):
    import torch
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
            and x.shape[0] > 0 and x.stride(-1) == 1)


def prequantized(layer):
    method = getattr(layer, "quant_method", None)
    return (type(method).__name__ == "SlimQuantW4A8Int8LinearMethod"
            and getattr(method, "supports_prequantized_input", False))


def around_apply(original, self, layer, x, *args, **kwargs):
    pending = _pending.get()
    if pending is not None and pending[0] is layer and pending[1] is x:
        _pending.set(None)
        kwargs["input_quant_args"] = pending[2]
    return original(self, layer, x, *args, **kwargs)


def around_mlp(original, self, hidden_states, *args, **kwargs):
    if (args or kwargs or not enabled("SGLANG_K3_FUSED_SITU_INT8")
            or not prequantized(self.down_proj)
            or not self.down_proj.input_is_parallel
            or type(self.act_fn).__name__ != "SituAndMul"
            or not supported(hidden_states)):
        if enabled("SGLANG_K3_FUSED_SITU_INT8"):
            status("situ_int8", "skip: INT8 projection, input, or activation contract")
        return original(self, hidden_states, *args, **kwargs)
    status("situ_int8", "eligible INT8 projection")
    width = self.down_proj.input_size_per_partition
    beta = self.act_fn.beta
    linear_beta = self.act_fn.linear_beta
    if not (0 < width <= 8192 and width % 16 == 0
            and beta > 0 and linear_beta is not None and linear_beta > 0):
        return original(self, hidden_states, *args, **kwargs)
    try:
        from lightop import fuse_situ_mul_quant_contiguous
    except ImportError:
        return original(self, hidden_states, *args, **kwargs)
    gate_up, _ = self.gate_up_proj(hidden_states)
    if not supported(gate_up):
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output
    status("situ_int8", "hit LightOp")
    x_q, x_scale = fuse_situ_mul_quant_contiguous(gate_up, beta, linear_beta)
    placeholder = gate_up[:, :gate_up.shape[1] // 2]
    token = _pending.set((self.down_proj, placeholder, (x_q, x_scale)))
    try:
        output, _ = self.down_proj(placeholder)
        if _pending.get() is not None:
            raise RuntimeError("Kimi31 SiTU prequantized consumer was not invoked")
        return output
    finally:
        _pending.reset(token)


def around_moe(original, self, *args, **kwargs):
    token = _pending.set(None)
    try:
        result = original(self, *args, **kwargs)
        if _pending.get() is not None:
            raise RuntimeError("Kimi31 latent prequantized consumer was not invoked")
        return result
    finally:
        _pending.reset(token)


def around_latent_norm(original, self, latent):
    import torch
    norm = self.routed_expert_norm
    layer = self.routed_expert_up_proj
    if (not enabled("SGLANG_K3_FUSED_LATENT_RMS_INT8") or norm is None
            or not prequantized(layer) or not supported(latent)
            or not 0 < latent.shape[-1] <= 8192 or latent.shape[-1] % 16
            or norm.weight.dtype != latent.dtype):
        if enabled("SGLANG_K3_FUSED_LATENT_RMS_INT8"):
            status("latent_rms_int8", "skip: INT8 projection or input contract")
        return original(self, latent)
    try:
        from lightop import rms_norm_dynamic_per_token_quant_sglang
    except ImportError:
        return original(self, latent)
    status("latent_rms_int8", "hit LightOp")
    quant = rms_norm_dynamic_per_token_quant_sglang(
        latent, norm.weight, epsilon=norm.variance_epsilon,
        quant_dtype=torch.int8, residual=None, update_input=False,
    )
    if _pending.get() is not None:
        raise RuntimeError("Kimi31 latent prequantized input was not consumed")
    _pending.set((layer, latent, quant))
    return latent


def around_attn_res(original, *args, **kwargs):
    if not enabled("SGLANG_K3_FUSED_ATTN_RES"):
        return original(*args, **kwargs)
    status("attention_residual", "hit HCU Triton")
    from hcu_sglang.kernels.kimi31.attn_res_hip import attn_res_hip
    return attn_res_hip(*args, **kwargs)


def around_mla_prepare(original, self, *args, **kwargs):
    if enabled("SGLANG_K3_FUSED_MLA_POST") and getattr(self, "use_dsa", False):
        status("mla_post", "skip: DSA structured/scaled cache is incompatible with plain 576-element MLA cache")
    if (not enabled("SGLANG_K3_FUSED_MLA_FRONT")
            or self.q_lora_rank is None or self.alt_stream is not None
            or not prequantized(self.q_b_proj)
            or getattr(self, "dcp_size", 1) != 1):
        if enabled("SGLANG_K3_FUSED_MLA_FRONT"):
            status("mla_front", "skip: INT8 q_b projection, stream, or DCP contract")
        return original(self, *args, **kwargs)
    state = {"owner": self, "packed": None}
    front_token = _mla_front.set(state)
    quant_token = _pending.set(None)
    try:
        result = original(self, *args, **kwargs)
        if _pending.get() is not None:
            raise RuntimeError("Kimi31 MLA prequantized consumer was not invoked")
        return result
    finally:
        _pending.reset(quant_token)
        _mla_front.reset(front_token)


def around_rms_norm(original, self, x, *args, **kwargs):
    state = _mla_front.get()
    if state is None or args or kwargs:
        return original(self, x, *args, **kwargs)
    owner = state["owner"]
    packed = state["packed"]
    if self is owner.kv_a_layernorm and packed is not None:
        expected = packed[:, owner.q_lora_rank:owner.q_lora_rank + owner.kv_lora_rank]
        if x.shape == expected.shape and x.data_ptr() == expected.data_ptr():
            return x
    if self is not owner.q_a_layernorm:
        return original(self, x, *args, **kwargs)
    packed = getattr(x, "_base", None)
    if packed is None or packed.ndim != 2 or packed.shape != (
        x.shape[0], owner.q_lora_rank + owner.kv_lora_rank + owner.qk_rope_head_dim
    ) or x.data_ptr() != packed.data_ptr():
        return original(self, x, *args, **kwargs)
    from hcu_sglang.layers.fused_rms_quant import (
        supports_fused_mla_qkv_a_rms_quant_input,
        fused_mla_qkv_a_rms_norm_per_token_quant,
    )
    if not supports_fused_mla_qkv_a_rms_quant_input(
        packed, owner.q_a_layernorm.weight, owner.kv_a_layernorm.weight
    ):
        return original(self, x, *args, **kwargs)
    quant = fused_mla_qkv_a_rms_norm_per_token_quant(
        packed_input=packed, q_weight=owner.q_a_layernorm.weight,
        kv_weight=owner.kv_a_layernorm.weight,
        q_epsilon=owner.q_a_layernorm.variance_epsilon,
        kv_epsilon=owner.kv_a_layernorm.variance_epsilon,
    )
    status("mla_front", "hit LightOp packed Q/KV RMS+INT8")
    state["packed"] = packed
    _pending.set((owner.q_b_proj, x, quant))
    return x


def around_hip_gate(original, hidden_size, nvb):
    if not enabled("SGLANG_K3_FUSED_ATTN_RES"):
        return original(hidden_size, nvb)
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(hidden_size, nvb)
    tile_h = 1 << (max(hidden_size, 1) - 1).bit_length()
    tile_r = 1 << (max(nvb, 1) - 1).bit_length()
    result = tile_h * tile_r <= 8 * 8192
    status("attention_residual_gate", "enabled" if result else "skip: register tile budget")
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    bindings = (
        ("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare", around_mla_prepare),
        ("sglang.srt.layers.attn_residual._use_hip_fused", around_hip_gate),
        ("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm.DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare", around_mla_prepare),
        ("sglang.srt.layers.layernorm.RMSNorm.forward", around_rms_norm),
        ("sglang.srt.models.kimi_k3.KimiK3MLP.forward", around_mlp),
        ("sglang.srt.models.kimi_k3.KimiK3MoE.forward", around_moe),
        ("sglang.srt.models.kimi_k3.KimiK3MoE._latent_norm", around_latent_norm),
        ("hcu_sglang.layers.quantization.slimquant_w4a8.SlimQuantW4A8Int8LinearMethod.apply", around_apply),
        ("sglang.kernels.ops.attention.attn_res_hip.attn_res_hip", around_attn_res),
    )
    for target, callback in bindings:
        HookRegistry.register(target, callback, HookType.AROUND)

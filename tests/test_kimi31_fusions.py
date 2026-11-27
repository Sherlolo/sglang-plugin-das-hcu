import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("kimi31_fusions_under_test", Path(__file__).parents[1] / "hcu_sglang/kimi31_fusions.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class FusionContractTests(unittest.TestCase):
    def test_situ_uses_quant_consumer_and_preserves_linear_owner(self):
        class Tensor:
            shape = (2, 64)
            def __getitem__(self, item):
                return placeholder
        class SlimQuantW4A8Int8LinearMethod:
            supports_prequantized_input = True
        class SituAndMul:
            beta = 4.0
            linear_beta = 25.0
        class Down:
            quant_method = SlimQuantW4A8Int8LinearMethod()
            input_is_parallel = True
            input_size_per_partition = 32
            def __call__(self, x):
                def apply(method, layer, value, **kwargs):
                    self.received = kwargs["input_quant_args"]
                    return "projected"
                return F.around_apply(apply, self.quant_method, self, x), None
        placeholder = object()
        down = Down()
        quant = (object(), object())
        owner = types.SimpleNamespace(down_proj=down, act_fn=SituAndMul(),
                                      gate_up_proj=lambda x: (Tensor(), None))
        provider = types.SimpleNamespace(fuse_situ_mul_quant_contiguous=lambda *a: quant)
        with patch.dict(os.environ, {"SGLANG_K3_FUSED_SITU_INT8": "1"}), \
             patch.dict(sys.modules, {"lightop": provider}), \
             patch.object(F, "supported", return_value=True):
            result = F.around_mlp(lambda *a: self.fail("unfused path"), owner, object())
        self.assertEqual(result, "projected")
        self.assertEqual(down.received, quant)
        self.assertIsNone(F._pending.get())

    def test_mla_front_preserves_normalized_q_for_indexer_and_quant_consumer(self):
        class Tensor:
            ndim = 2
            def __init__(self, shape, pointer, base=None):
                self.shape, self.pointer, self._base = shape, pointer, base
            def data_ptr(self):
                return self.pointer
            def __getitem__(self, index):
                return kv
        packed = Tensor((2, 2112), 100)
        q = Tensor((2, 1536), 100, packed)
        kv = Tensor((2, 512), 3172, packed)
        qnorm = types.SimpleNamespace(weight=object(), variance_epsilon=1e-6)
        knorm = types.SimpleNamespace(weight=object(), variance_epsilon=1e-6)
        owner = types.SimpleNamespace(q_lora_rank=1536, kv_lora_rank=512,
            qk_rope_head_dim=64, q_a_layernorm=qnorm, kv_a_layernorm=knorm,
            q_b_proj=object(), alt_stream=None, dcp_size=1)
        quant = (object(), object())
        calls = []
        provider = types.SimpleNamespace(
            supports_fused_mla_qkv_a_rms_quant_input=lambda *a: True,
            fused_mla_qkv_a_rms_norm_per_token_quant=lambda **kw: calls.append(kw) or quant)
        def prepare(self):
            qout = F.around_rms_norm(lambda *a: self.fail("duplicate q norm"), qnorm, q)
            kout = F.around_rms_norm(lambda *a: self.fail("duplicate kv norm"), knorm, kv)
            self.assertIs(qout, q)
            self.assertIs(kout, kv)
            result = F.around_apply(lambda *a, **kw: kw["input_quant_args"], None, owner.q_b_proj, qout)
            self.assertIs(result, quant)
            return "prepared"
        with patch.dict(os.environ, {"SGLANG_K3_FUSED_MLA_FRONT": "1"}), \
             patch.object(F, "prequantized", return_value=True), \
             patch.dict(sys.modules, {"hcu_sglang.layers.fused_rms_quant": provider}):
            result = F.around_mla_prepare(lambda owner: prepare(self), owner)
        self.assertEqual(result, "prepared")
        self.assertIs(calls[0]["packed_input"], packed)
        self.assertIsNone(F._pending.get())
        self.assertIsNone(F._mla_front.get())

    def test_physical_hcu_gate_overrides_cuda_semantics_only_when_enabled(self):
        provider = types.SimpleNamespace(is_hcu=lambda: True)
        with patch.dict(sys.modules, {"hcu_sglang.utils.detection": provider}), \
             patch.dict(os.environ, {"SGLANG_K3_FUSED_ATTN_RES": "1"}):
            self.assertTrue(F.around_hip_gate(lambda *a: False, 7168, 8))
            self.assertFalse(F.around_hip_gate(lambda *a: True, 7168, 9))
        with patch.dict(os.environ, {"SGLANG_K3_FUSED_ATTN_RES": "0"}):
            self.assertFalse(F.around_hip_gate(lambda *a: False, 7168, 8))

    def test_disabled_mlp_retains_previous_wrapper(self):
        calls = []
        with patch.dict(os.environ, {"SGLANG_K3_FUSED_SITU_INT8": "0"}):
            result = F.around_mlp(lambda *a, **k: calls.append((a, k)) or 19, object(), object())
        self.assertEqual(result, 19)
        self.assertEqual(len(calls), 1)

    def test_quant_consumer_is_identity_scoped_and_single_use(self):
        layer, other_layer, x, quant = object(), object(), object(), (object(), object())
        calls = []
        def apply(self, layer, x, **kwargs):
            calls.append(kwargs)
            return x
        token = F._pending.set((layer, x, quant))
        try:
            F.around_apply(apply, None, other_layer, x)
            self.assertEqual(calls[-1], {})
            self.assertIsNotNone(F._pending.get())
            F.around_apply(apply, None, layer, x)
            self.assertIs(calls[-1]["input_quant_args"], quant)
            self.assertIsNone(F._pending.get())
            F.around_apply(apply, None, layer, x)
            self.assertEqual(calls[-1], {})
        finally:
            F._pending.reset(token)

    def test_scope_restores_state_after_consumer_failure(self):
        old = object()
        token = F._pending.set(old)
        def failing(*args):
            F._pending.set((object(), object(), ()))
            raise ValueError("vendor failed")
        try:
            with self.assertRaisesRegex(ValueError, "vendor failed"):
                F.around_moe(failing, object())
            self.assertIs(F._pending.get(), old)
        finally:
            F._pending.reset(token)

    def test_unconsumed_quant_is_rejected(self):
        def leave_pending(*args):
            F._pending.set((object(), object(), ()))
        with self.assertRaisesRegex(RuntimeError, "consumer was not invoked"):
            F.around_moe(leave_pending, object())
        self.assertIsNone(F._pending.get())

    def test_unquantized_method_cannot_consume_int8(self):
        layer = types.SimpleNamespace(quant_method=types.SimpleNamespace(supports_prequantized_input=True))
        self.assertFalse(F.prequantized(layer))


if __name__ == "__main__":
    unittest.main()

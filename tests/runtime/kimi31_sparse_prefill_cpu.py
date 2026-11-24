import ast
import json
from pathlib import Path
import types
from typing import Optional
import unittest
from unittest.mock import patch

import torch
import sglang

from hcu_sglang import kimi31_sparse_prefill as bridge


def original_method():
    source = (Path(sglang.__file__).resolve().parent /
              'srt/layers/attention/dsa_backend.py').read_text()
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'DeepseekSparseAttnBackend')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == '_forward_flashmla_sparse')
    namespace = dict(torch=torch, Optional=Optional)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original sparse prefill>', 'exec'),
         namespace)
    return namespace['_forward_flashmla_sparse']


class SparsePrefillTests(unittest.TestCase):
    def test_backend_resolution_preserves_inputs(self):
        from sglang.srt.arg_groups.model_override_base import ResolvedView

        inputs = types.SimpleNamespace(dsa_prefill_backend='flashmla_sparse',
                                      dsa_decode_backend='flashmla_kv',
                                      enable_hisparse=False)
        calls = []

        def original(view):
            calls.append(view.dsa_prefill_backend)
            self.assertEqual(view.dsa_decode_backend, 'flashmla_kv')
            return {}

        with patch.object(torch.version, 'hip', 'test'):
            result = bridge.around_kimi_backends(original, ResolvedView(inputs))
        self.assertEqual(calls, ['flashmla_kv'])
        self.assertEqual(result, {'dsa_prefill_backend': 'flashmla_sparse'})
        self.assertEqual(inputs.dsa_prefill_backend, 'flashmla_sparse')

    def test_backend_resolution_rejects_hisparse(self):
        inputs = types.SimpleNamespace(dsa_prefill_backend='flashmla_sparse',
                                      dsa_decode_backend='flashmla_kv',
                                      enable_hisparse=True)
        with patch.object(torch.version, 'hip', 'test'):
            with self.assertRaisesRegex(ValueError, 'HiSparse'):
                bridge.around_kimi_backends(None, inputs)

    def run_case(self, heads, with_sink):
        original = original_method()
        q = torch.arange(3 * heads * 576).reshape(3, heads, 576).remainder(17).bfloat16()
        kv = torch.zeros(7, 1, 576, dtype=torch.bfloat16)
        table = torch.tensor([[1, -1], [2, 3], [0, 6]], dtype=torch.int32)
        lengths = torch.tensor([1, 2, 2], dtype=torch.int32)
        sink = torch.arange(heads).float() if with_sink else None
        sink_before = None if sink is None else sink.clone()
        calls = []

        def native(**kwargs):
            padded = ((heads + 63) // 64) * 64
            self.assertEqual(kwargs['q'].shape, (3, padded, 576))
            torch.testing.assert_close(kwargs['q'][:, :heads], q)
            self.assertTrue(torch.all(kwargs['q'][:, heads:] == 0))
            self.assertIs(kwargs['kv'], kv)
            torch.testing.assert_close(kwargs['indices'][:, 0], table)
            self.assertIs(kwargs['topk_length'], lengths)
            if with_sink:
                torch.testing.assert_close(kwargs['attn_sink'][:heads], sink)
                self.assertTrue(torch.all(kwargs['attn_sink'][heads:] == 0))
            else:
                self.assertIsNone(kwargs['attn_sink'])
            calls.append(kwargs)
            return kwargs['q'][..., :512].clone(), None, None

        backend = types.SimpleNamespace(device_sm_major=9, sparse_prefill_small_heads=False,
                                        _sink_pad_cache={}, _call_flash_mla_sparse_fwd=native)
        with self.assertRaisesRegex(AssertionError, 'cannot be padded'):
            original(backend, q, kv, 512, table, 0.125, topk_length=lengths,
                     attn_sink=sink, allow_small_heads=True)
        self.assertEqual(calls, [])
        with patch.object(torch.version, 'hip', 'cpu-fixture'):
            with patch.object(torch.Tensor, 'is_cuda', property(lambda tensor: True)):
                output = bridge.around_sparse_prefill(
                    original, backend, q, kv, 512, table, 0.125, lengths, sink, True)
        self.assertEqual(output.shape, (3, heads, 512))
        torch.testing.assert_close(output, q[..., :512])
        self.assertEqual(len(calls), 1)
        if sink is not None:
            torch.testing.assert_close(sink, sink_before)

    def test_h96_preserves_sink_and_trims_native_h128(self):
        self.run_case(96, True)

    def test_h12_preserves_indices_and_trims_native_h64(self):
        self.run_case(12, False)

    def test_cpu_fallback_keeps_current_original_chain(self):
        sentinel = object()
        q = torch.zeros(1, 96, 576, dtype=torch.bfloat16)
        captured = []
        def original(*args, **kwargs):
            captured.append((args, kwargs))
            return sentinel
        self.assertIs(bridge.around_sparse_prefill(original, None, q, q, 512, None, 1), sentinel)
        self.assertIs(captured[0][0][1], q)

    def test_native_errors_propagate(self):
        def original(*args, **kwargs):
            raise RuntimeError('native sparse failure')
        with self.assertRaisesRegex(RuntimeError, 'native sparse failure'):
            bridge._forward_padded_sparse(original, None,
                torch.zeros(1, 96, 576, dtype=torch.bfloat16), None, 512, None, 1)

    def test_registration_matches_optional_profile(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        contract = json.loads((Path(__file__).resolve().parents[1] / 'contracts' /
                               'hcu_source_alignment_targets.json').read_text())
        profile = contract['optional_profiles']['kimi31_flashmla_sparse_heads']
        with patch.dict('os.environ', profile['environment']):
            with patch.object(HookRegistry, 'register') as register:
                bridge.register_hooks()
        self.assertEqual([call.args[0] for call in register.call_args_list],
                         profile['expected_hook_targets'])
        self.assertIs(register.call_args.args[1], bridge.around_sparse_prefill)
        self.assertEqual(register.call_args.args[2], HookType.AROUND)
        with patch.dict('os.environ', {'HCU_KIMI31_FLASHMLA_SPARSE_HEADS': '0'}):
            with patch.object(HookRegistry, 'register') as register:
                bridge.register_hooks()
        register.assert_not_called()


if __name__ == '__main__':
    unittest.main()

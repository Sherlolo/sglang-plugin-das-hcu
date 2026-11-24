import sys
import types
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.kimi31_mqa import around_paged_mqa
from hcu_sglang import kimi31_mqa as bridge


class RaggedPrefill(unittest.TestCase):
    def test_partial_pages_preserve_signed_bytes_scales_and_padding(self):
        k = torch.arange(70 * 256).reshape(70, 256).remainder(15).sub(7).float().to(torch.float8_e4m3fn)[:, ::2]
        scales = torch.arange(140).float()[::2] + 0.25
        packed = bridge._pack_ragged_k(k, scales)
        pages = packed.reshape(2, 64 * 132)
        payload = pages[:, :8192].reshape(128, 128)
        recovered = payload.view(torch.float8_e4m3fn)
        recovered_scales = pages[:, 8192:].contiguous().view(torch.float32).flatten()
        self.assertTrue(torch.equal(recovered[:70].float(), k.float()))
        self.assertTrue(torch.equal(recovered_scales[:70], scales))
        self.assertTrue(torch.all(payload[70:] == 0))
        self.assertTrue(torch.all(recovered_scales[70:] == 0))

    def test_prefix_native_call_keeps_ragged_lower_bounds(self):
        torch.manual_seed(42)
        q = torch.randn(4, 8, 128).to(torch.float8_e4m3fn)
        k = torch.randn(70, 128).to(torch.float8_e4m3fn)
        scales = torch.rand(70) + 0.1
        weights = torch.randn(4, 8)
        ks = torch.tensor([0, 7, 17, 70], dtype=torch.int64)
        ke = torch.tensor([0, 70, 17, 70], dtype=torch.int64)
        seen = []

        def native(queries, packed, gates, lens, table, width, **kwargs):
            self.assertTrue(kwargs['is_prefill'])
            self.assertTrue(kwargs['clean_logits'])
            self.assertEqual(queries.shape, (4, 1, 8, 128))
            self.assertEqual(lens.dtype, torch.int32)
            self.assertEqual(table.dtype, torch.int32)
            self.assertTrue(torch.equal(table, torch.tensor([[0, 1]]).expand(4, -1)))
            raw = packed.reshape(2, -1)
            keys = raw[:, :8192].reshape(128, 128).view(torch.float8_e4m3fn).float()[:width]
            sf = raw[:, 8192:].contiguous().view(torch.float32).reshape(-1)[:width]
            dots = torch.einsum('qhd,kd->qhk', queries[:, 0].float(), keys)
            result = (dots.relu() * gates[:, :, None]).sum(1) * sf[None, :]
            result.masked_fill_(torch.arange(width)[None, :] >= lens[:, None], float('-inf'))
            seen.append(True)
            return result

        with patch.dict(sys.modules, deepgemm=types.SimpleNamespace(paged_mqa_logits=native)):
            actual = bridge._ragged_mqa_logits(q, (k, scales), weights, ks, ke)
        reference = (torch.einsum('qhd,kd->qhk', q.float(), k.float()).relu() * weights[:, :, None]).sum(1) * scales[None, :]
        valid = (torch.arange(70)[None, :] >= ks[:, None]) & (torch.arange(70)[None, :] < ke[:, None])
        reference.masked_fill_(~valid, float('-inf'))
        torch.testing.assert_close(actual, reference)
        self.assertEqual(seen, [True])

    def test_empty_inputs_do_not_call_native(self):
        for rows, width in [(0, 70), (4, 0)]:
            with patch.dict(sys.modules, deepgemm=types.SimpleNamespace()):
                actual = bridge._ragged_mqa_logits(torch.empty(rows, 8, 128),
                    (torch.empty(width, 128), torch.empty(width)), torch.empty(rows, 8),
                    torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32))
            self.assertEqual(actual.shape, (rows, width))

    def test_unsupported_compute_preserves_current_original(self):
        marker = object()
        args = (object(), torch.empty(1, 8, 128), (torch.empty(1, 128), torch.ones(1)),
                torch.ones(1, 8), torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32))
        calls = []
        def original(*values):
            calls.append(values)
            return marker
        self.assertIs(bridge.around_ragged_mqa(original, *args), marker)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(a is b for a, b in zip(calls[0], args)))

    def test_native_failures_are_not_hidden_by_fallback(self):
        def native(*args, **kwargs):
            raise RuntimeError('native failure')
        with patch.dict(sys.modules, deepgemm=types.SimpleNamespace(paged_mqa_logits=native)):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                bridge._ragged_mqa_logits(torch.empty(1, 8, 128).to(torch.float8_e4m3fn),
                    (torch.empty(70, 128).to(torch.float8_e4m3fn), torch.ones(70)), torch.ones(1, 8),
                    torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32))

    def test_prefill_registration_is_independent_from_decode_flag(self):
        import json
        from pathlib import Path
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        contract = json.loads((Path(__file__).resolve().parents[1] / 'contracts' /
                               'hcu_source_alignment_targets.json').read_text())
        profile = contract['optional_profiles']['kimi31_deepgemm_prefill_mqa']
        with patch.dict('os.environ', profile['environment']):
            with patch.object(HookRegistry, 'register') as register:
                bridge.register_hooks()
        register.assert_called_once_with('sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._mqa_logits',
                                         bridge.around_ragged_mqa, HookType.AROUND)
        self.assertEqual([call.args[0] for call in register.call_args_list],
                         profile['expected_hook_targets'])


class CausalRows(unittest.TestCase):
    def run_case(self, explicit):
        batch, rows, heads = 2, 8, 32
        q = torch.zeros(batch, rows, heads, 128).to(torch.float8_e4m3fn)
        cache = torch.empty(4, 64, 1, 132, dtype=torch.uint8)
        weights = torch.ones(batch * rows, heads)
        end = torch.tensor([50, 70], dtype=torch.int32)
        expected = (end[:, None] - rows + 1 + torch.arange(rows, dtype=torch.int32)).flatten()
        contexts = expected.view(batch, rows) if explicit else end[:, None]
        table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        calls = []
        def native(q1, c1, w1, lens, pages, max_len, **kwargs):
            self.assertEqual(q1.shape, (batch * rows, 1, heads, 128))
            self.assertTrue(torch.equal(lens, expected))
            self.assertTrue(torch.equal(pages, table.repeat_interleave(rows, 0)))
            self.assertTrue(kwargs['clean_logits'])
            self.assertIs(c1, cache)
            calls.append(True)
            return torch.empty(batch * rows, max_len)
        with patch.dict(sys.modules, deepgemm=types.SimpleNamespace(paged_mqa_logits=native)):
            out = around_paged_mqa(None, None, q, cache, weights, contexts, table, None, 128)
        self.assertEqual(out.shape, (16, 128))
        self.assertEqual(len(calls), 1)

    def test_explicit_causal_rows(self):
        self.run_case(True)

    def test_request_endpoints(self):
        self.run_case(False)

    def test_fallback_preserves_original(self):
        sentinel = object()
        def original(*args, **kwargs):
            return sentinel
        q = torch.empty(2, 1, 32, 128, dtype=torch.bfloat16)
        cache = torch.empty(4, 64, 1, 132, dtype=torch.uint8)
        self.assertIs(around_paged_mqa(original, None, q, cache, None, None, None, None, 128), sentinel)


class SharedTP16(unittest.TestCase):
    def test_fp32_reduction_and_bf16_output(self):
        from hcu_sglang.kimi31_shared_tp import around_reduce_scatter
        testcase = self
        class Group:
            world_size = 16
            def reduce_scatter_tensor(self, output, values):
                testcase.assertEqual(output.dtype, torch.float32)
                testcase.assertEqual(values.dtype, torch.float32)
                output.copy_(values.reshape(16, 2, 4).sum(0))
        layer = types.SimpleNamespace(_shared_experts_tp_group=Group())
        hidden = torch.empty(2, 4, dtype=torch.bfloat16)
        values = torch.arange(128).reshape(32, 4).bfloat16()
        result = around_reduce_scatter(None, layer, values, hidden)
        self.assertEqual(result.dtype, hidden.dtype)
        self.assertTrue(torch.equal(result, values.float().reshape(16, 2, 4).sum(0).bfloat16()))

    def test_other_groups_keep_original(self):
        from hcu_sglang.kimi31_shared_tp import around_reduce_scatter
        layer = types.SimpleNamespace(_shared_experts_tp_group=types.SimpleNamespace(world_size=4))
        sentinel = object()
        self.assertIs(around_reduce_scatter(lambda *a: sentinel, layer,
                      torch.empty(8, 4, dtype=torch.bfloat16), torch.empty(2, 4)), sentinel)


if __name__ == '__main__':
    unittest.main()

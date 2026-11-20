import sys
import types
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.kimi31_mqa import around_paged_mqa


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

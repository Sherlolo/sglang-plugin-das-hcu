import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hcu_sglang import kimi31_dcp as bridge


class Pool:
    __module__ = 'sglang.srt.mem_cache.memory_pool'


class Inventory(unittest.TestCase):
    def test_dspark_triton_draft_uses_hip_replication_for_all_consumers(self):
        from sglang.srt.configs.attention_execution import attention_dcp_size
        args = dict(backend='triton', model_config=None, dcp_size=16,
                    is_draft_worker=True, is_dspark=True, is_hip=False)
        with patch.object(torch.version, 'hip', 'test'):
            self.assertEqual(bridge.around_attention_dcp_size(attention_dcp_size, **args), 1)
            self.assertEqual(bridge.around_attention_dcp_size(attention_dcp_size,
                **dict(args, is_draft_worker=False)), 16)
            self.assertEqual(bridge.around_attention_dcp_size(attention_dcp_size,
                **dict(args, is_dspark=False)), 16)
        with patch.object(torch.version, 'hip', None):
            self.assertEqual(bridge.around_attention_dcp_size(attention_dcp_size, **args), 16)

    def test_hcu_candidate_exchange_declines_nvlink_without_calling_nvml(self):
        calls = []
        def original(*args, **kwargs):
            calls.append(args)
            return True
        with patch.object(torch.version, 'hip', 'test'):
            self.assertFalse(bridge.around_nvlink_clique(original, object()))
            self.assertEqual(calls, [])
        with patch.object(torch.version, 'hip', None):
            self.assertTrue(bridge.around_nvlink_clique(original, 'group'))
            self.assertEqual(calls, [('group',)])

    def test_fake_decode_validation_overlay_preserves_actual_transport(self):
        def provider(args, config):
            return args.disaggregation_transfer_backend
        provider.__module__ = 'sglang.srt.arg_groups.model_overrides.kimi_k3'
        provider.__name__ = '_kimi_k31_overrides'
        args = SimpleNamespace(dcp_size=16, disaggregation_mode='decode',
                               disaggregation_transfer_backend='fake')
        def invoke(fn, inputs, config):
            return fn(inputs, config)
        with patch.dict(os.environ, HCU_KIMI31_DCP_FAKE='1'), patch.object(torch.version, 'hip', 'test'):
            self.assertEqual(bridge.around_model_provider(invoke, provider, args, None), 'mooncake')
            self.assertEqual(args.disaggregation_transfer_backend, 'fake')
            for field, value in [('dcp_size', 1), ('disaggregation_mode', 'prefill'),
                                 ('disaggregation_transfer_backend', 'nixl')]:
                inputs = SimpleNamespace(**dict(vars(args), **{field: value}))
                seen = []
                bridge.around_model_provider(lambda f, a, c: seen.append(a), provider, inputs, None)
                self.assertIs(seen[0], inputs)

    def test_physical_storage_counts_views_once_and_keeps_roles_separate(self):
        pool = Pool()
        pool.kv_buffer = [torch.zeros(64, 128, dtype=torch.uint8)]
        pool.alias = pool.kv_buffer[0].view(-1)
        pool.parent = pool
        slots_type = type('SlotsPool', (), {'__module__': Pool.__module__, '__slots__': ()})
        pool.slots = slots_type()
        pool.index = torch.zeros(64, 132, dtype=torch.uint8)
        runner = SimpleNamespace(token_to_kv_pool=pool, is_draft_worker=False,
            max_total_num_tokens=1024, max_running_requests=16,
            logical_max_total_num_tokens=16384,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1024))
        parallel = SimpleNamespace(attn_dcp_size=16, attn_dcp_rank=3)
        from sglang.srt import runtime_context
        with TemporaryDirectory() as directory, patch.dict(os.environ, HCU_KIMI31_KV_INVENTORY=directory), \
             patch.object(runtime_context, 'get_parallel', return_value=parallel), \
             patch.object(torch.distributed, 'get_rank', return_value=3), \
             patch.object(torch.cuda, 'memory_allocated', return_value=123), \
             patch.object(torch.cuda, 'memory_reserved', return_value=456):
            sentinel = object()
            self.assertIs(bridge.after_alloc_memory_pool(sentinel, runner), sentinel)
            record = json.loads((Path(directory) / 'target-rank3.json').read_text())
            self.assertEqual(record['pool_storage_bytes'], 64 * (128 + 132))
            self.assertEqual(len(record['tensors']), 2)
            self.assertEqual(record['allocator_page_size'], 1024)
            runner.is_draft_worker = True
            bridge.after_alloc_memory_pool(None, runner)
            self.assertTrue((Path(directory) / 'draft-rank3.json').is_file())

    def test_disabled_inventory_does_not_access_runner(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(bridge.after_alloc_memory_pool(17, object()), 17)


if __name__ == '__main__':
    unittest.main()

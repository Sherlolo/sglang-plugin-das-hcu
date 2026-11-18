# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import ast
import __future__
from dataclasses import dataclass
from inspect import signature
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
import types
import unittest
from unittest.mock import patch
if sys.flags.no_site:sys.path.append(sysconfig.get_paths()['purelib'])
try:
    import torch
    import torch.nn.functional as F
except ImportError:raise SystemExit(77)
REPO=Path(__file__).resolve().parents[1]
SGLANG=Path(os.environ.get('KIMI_SGLANG_SOURCE',REPO.parent/'sglang'))
def package(name):
    if name not in sys.modules:
        value=types.ModuleType(name);value.__path__=[];sys.modules[name]=value
    if '.' in name:
        parent,child=name.rsplit('.',1)
        setattr(package(parent),child,sys.modules[name])
    return sys.modules[name]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
for name in ['sglang','sglang.srt','hcu_sglang','hcu_sglang.layers','hcu_sglang.layers.moe','lightop']:
    package(name)
runtime=package('sglang.srt.runtime_context');runtime.get_parallel=lambda:None
common=package('sglang.srt.utils');common.get_bool_env_var=lambda *args,**kwargs:True
CT=load('hcu_sglang.layers.moe.ct_w4a8',REPO/'hcu_sglang/layers/moe/ct_w4a8.py')
COMPAT=load('hcu_sglang.kimi31_graph_compat',REPO/'hcu_sglang/kimi31_graph_compat.py')

class DeepEPMaskedTests(unittest.TestCase):
    def test_masked_wrappers_return_output_not_tile_metadata(self):
        output = torch.empty((2, 3))
    
        def gemm(a, b, d, mask, hint):
            d.fill_(7)
            return (64, 64)
        vendor = types.SimpleNamespace(m_grouped_w4a8_gemm_nt_masked=gemm, m_grouped_w4a8_gemm_nt_masked_hipc=gemm)
        with patch.dict(sys.modules, {'deepgemm': vendor}):
            for fn in [CT._masked, CT._masked_hipc]:
                actual = fn(output, output, output, output, output, torch.zeros(2, dtype=torch.int32), 1)
                self.assertIs(actual, output)
                self.assertTrue(torch.all(output == 7))

    def test_masked_situ_preserves_counts_and_parameters(self):
        calls = []
    
        def gemm(x, scale, weight, wscale, out, mask, hint):
            calls.append((mask, hint))
            out.zero_()
    
        def situ(x, mask, beta, linear, expect_m):
            self.assertEqual((beta, linear, expect_m), (4.0, 25.0, 2))
            return (torch.zeros(2, 4, 2, dtype=torch.int8), torch.ones(2, 4, 1))
        lightop = package('lightop')
        lightop.fuse_silu_mul_quant_ep = lambda *a, **kw: None
        lightop.fuse_silu_mul_clamp_quant_ep = lambda *a, **kw: None
        lightop.fuse_situ_mul_quant_ep = situ
        layer = types.SimpleNamespace(moe_runner_config=types.SimpleNamespace(activation='situ', gemm1_alpha=4.0, gemm1_clamp_limit=25.0, swiglu_limit=None), w13_weight_scale=torch.ones(2, 4, 1), w2_weight_scale=torch.ones(2, 4, 1), w13_weight=torch.zeros(2, 4, 4), w2_weight=torch.zeros(2, 4, 2), w4a8_padded_intermediate_size=2)
        mask = torch.tensor([2, 0], dtype=torch.int32)
        output = types.SimpleNamespace(hidden_states=torch.zeros(2, 4, 4, dtype=torch.int8), hidden_states_scale=torch.ones(2, 4, 1), masked_m=mask, expected_m=2)
        with patch.object(torch.ops.sglang, 'hcu_ct_w4a8_masked_hipc', gemm, create=True):
            result = CT.forward_masked(layer, output)
        self.assertEqual(result.shape, (2, 4, 4))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all((value[0] is mask for value in calls)))

    def test_hcu_metadata_rejects_geometry_drift_and_preserves_tensor_contract(self):
        first = COMPAT.SparseFlashMLAMetadata(types.SimpleNamespace(have_initialized=False), (1, 64, 2048))
        second = COMPAT.SparseFlashMLAMetadata(types.SimpleNamespace(have_initialized=False), (1, 64, 2048))
        first.copy_(second)
        self.assertIs(first.flashmla_metadata, second.flashmla_metadata)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            first.copy_(COMPAT.SparseFlashMLAMetadata(types.SimpleNamespace(have_initialized=False), (2, 64, 2048)))
        with self.assertRaises(ValueError):
            first.slice(slice(1, 2))
        tensor_metadata = types.SimpleNamespace(flashmla_metadata=torch.ones(2, 4), num_splits=torch.ones(3))
        self.assertIs(COMPAT.around_compute_flashmla_metadata(lambda *a, **kw: tensor_metadata, None, torch.ones(2), 1), tensor_metadata)

    def test_hcu_static_sparse_schedule_rejects_dynamic_overrides(self):
        scheduler = types.SimpleNamespace(have_initialized=False)
        COMPAT.SparseFlashMLAMetadata(scheduler, (2, 64, 2048))
        kwargs = {'tile_scheduler_metadata': scheduler, 'indices': torch.zeros(2, 1, 2048), 'is_fp8_kvcache': True, 'num_splits': None}
        COMPAT.validate_sparse_scheduler(kwargs)
        for change in [{'indices': None}, {'is_fp8_kvcache': False}, {'topk_length': torch.ones(2)}, {'extra_topk_length': torch.ones(2)}, {'extra_k_cache': torch.ones(2)}, {'num_splits': torch.ones(1)}]:
            with self.assertRaises(ValueError):
                COMPAT.validate_sparse_scheduler({**kwargs, **change})
        COMPAT.validate_sparse_scheduler({'tile_scheduler_metadata': object()})

    def test_day0_flashmla_public_wrapper_validates_scheduler_before_native_call(self):
        tree = ast.parse((REPO / 'hcu_sglang/kimi31_day0.py').read_text())
        factory = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_flashmla_operation'))
        calls = []
    
        def native(**kwargs):
            calls.append(kwargs)
            return 'native-result'
        ns = {'signature': signature, 'get_flashmla_op': lambda *a, **kw: native}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), 'actual_day0_flashmla_wrapper', 'exec'), ns)
        wrapped = ns[factory.name]('flash_mla_with_kvcache')
        scheduler = types.SimpleNamespace(have_initialized=False)
        COMPAT.SparseFlashMLAMetadata(scheduler, (2, 64, 2048))
        kwargs = {'tile_scheduler_metadata': scheduler, 'indices': torch.zeros(2, 1, 2048), 'is_fp8_kvcache': True, 'num_splits': None, 'kv_format': 'V32'}
        self.assertEqual(wrapped(None, **kwargs), 'native-result')
        self.assertNotIn('kv_format', calls[0])
        with self.assertRaises(ValueError):
            wrapped(None, **{**kwargs, 'topk_length': torch.ones(2)})
        self.assertEqual(len(calls), 1)
        register = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'register_hooks'))
        self.assertTrue(any((isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'register_graph_compat') for n in ast.walk(register))))

@unittest.skipUnless((SGLANG/'python/sglang/srt/plugins/hook_registry.py').is_file(),'Set KIMI_SGLANG_SOURCE to validate upstream callers')
class DeepEPCallerTests(unittest.TestCase):
    def actual_method(self, relative, owner, name, namespace):
        tree = ast.parse((SGLANG / 'python/sglang' / relative).read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner))
        node = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec', flags=__future__.annotations.compiler_flag), namespace)
        return namespace[name]

    def test_kda_packed_decode_preserves_state_indices_without_host_sync(self):
        called = []
    
        def kernel(**kwargs):
            called.append(kwargs)
            kwargs['out'].zero_()
            state = kwargs['initial_state']
            indices = kwargs['ssm_state_indices']
            state.index_add_(0, indices, torch.ones((len(indices), 2, 2, 2)))
        method = self.actual_method('srt/layers/attention/linear/kernels/kda_triton.py', 'TritonKDAKernel', 'packed_decode', {'torch': torch, 'fused_recurrent_kda_packed_decode': kernel})
        for size in [1, 2, 4, 8]:
            state = torch.zeros(16, 2, 2, 2)
            indices = torch.arange(size, dtype=torch.int64) * 2
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host item')), patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host cpu')), patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host list')):
                output = method(None, torch.ones(size, 12), torch.ones(size, 2, 2), torch.ones(size, 1, 2), A_log=torch.zeros(2), dt_bias=torch.zeros(2, 2), scale=0.5, ssm_states=state, cache_indices=indices, num_v_heads=2, head_v_dim=2, lower_bound=-5.0)
            self.assertEqual(output.shape, (1, size, 2, 2))
            self.assertIs(called[-1]['initial_state'], state)
            self.assertIs(called[-1]['ssm_state_indices'], indices)
            self.assertEqual(called[-1]['lower_bound'], -5.0)
            expected = torch.zeros_like(state)
            expected[indices] = 1
            torch.testing.assert_close(state, expected)

    def test_upstream_zero_row_kda_requires_plugin_guard(self):
        method = self.actual_method('srt/layers/attention/linear/kernels/kda_triton.py', 'TritonKDAKernel', 'packed_decode', {'torch': torch, 'fused_recurrent_kda_packed_decode': lambda **kw: kw['out'].zero_()})
        with self.assertRaisesRegex(RuntimeError, 'reshape'):
            method(None, torch.zeros(0, 12), torch.zeros(0, 2, 2), torch.zeros(0, 1, 2), A_log=torch.zeros(2), dt_bias=torch.zeros(2, 2), scale=0.5, ssm_states=torch.zeros(16, 2, 2, 2), cache_indices=torch.zeros(0, dtype=torch.int64), num_v_heads=2, head_v_dim=2, lower_bound=-5.0)

    def test_kda_nonpacked_decode_retains_safe_gate_and_state(self):
        calls = []
        method = self.actual_method('srt/layers/attention/linear/kernels/kda_triton.py', 'TritonKDAKernel', 'decode', {'torch': torch, 'fused_sigmoid_gating_delta_rule_update': lambda **kw: calls.append(kw) or 'output'})
        values = [torch.zeros(1, 2, 2, 2) for _ in range(5)]
        state = torch.zeros(8, 2, 2, 2)
        indices = torch.tensor([1, 4])
        starts = torch.tensor([0, 1, 2])
        self.assertEqual(method(None, *values, A_log=torch.zeros(2), dt_bias=torch.zeros(2, 2), ssm_states=state, cache_indices=indices, query_start_loc=starts, lower_bound=-5.0), 'output')
        self.assertIs(calls[0]['initial_state_source'], state)
        self.assertIs(calls[0]['initial_state_indices'], indices)
        self.assertIs(calls[0]['cu_seqlens'], starts)
        self.assertTrue(calls[0]['is_kda'])
        self.assertEqual(calls[0]['lower_bound'], -5.0)

    def test_hybrid_decode_routes_kda_and_mla_without_host_sync(self):
        method = self.actual_method('srt/layers/attention/hybrid_linear_attn_backend.py', 'HybridLinearAttnBackend', 'forward_decode', {})
        calls = []
        owner = types.SimpleNamespace(_is_full_attn=lambda layer, index: layer == 'mla', full_attn_backend=types.SimpleNamespace(forward_decode=lambda *a, **kw: calls.append(('mla', a, kw)) or 'mla'), linear_attn_backend=types.SimpleNamespace(forward_decode=lambda **kw: calls.append(('kda', (), kw)) or 'kda'))
        for name in ['mla', 'kda']:
            self.assertEqual(method(owner, name, 'batch', mixed_qkv='qkv', a='a', b='b'), name)
        self.assertEqual([call[0] for call in calls], ['mla', 'kda'])
        self.assertEqual(calls[1][2]['mixed_qkv'], 'qkv')

    def test_upstream_flashmla_object_requires_plugin_contract(self):
        tree = ast.parse((SGLANG / 'python/sglang/srt/layers/attention/dsa_backend.py').read_text())
        node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DSAFlashMLAMetadata'))
        ns = {'dataclass': dataclass, 'torch': torch, '__name__': __name__}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'upstream_flashmla_metadata', 'exec', flags=__future__.annotations.compiler_flag), ns)
        value = ns[node.name](object(), None)
        with self.assertRaises(TypeError):
            value.slice(slice(0, 2)).copy_(ns[node.name](object(), None))

    def test_hcu_graph_metadata_keeps_scheduler_and_device_buffers_per_bucket(self):
        method = self.actual_method('srt/layers/attention/dsa_backend.py', 'DeepseekSparseAttnBackend', '_compute_flashmla_metadata', {'DSAFlashMLAMetadata': lambda **kw: types.SimpleNamespace(**kw)})
        owner = types.SimpleNamespace(flashmla_kv_num_q_heads=64, dsa_index_topk=2048, _call_get_mla_metadata=lambda **kw: (types.SimpleNamespace(have_initialized=False, tile_scheduler_metadata=None, num_splits=None), None))
        owner._compute_flashmla_metadata = lambda **kw: COMPAT.around_compute_flashmla_metadata(method, owner, **kw)
        tree = ast.parse((SGLANG / 'python/sglang/srt/layers/attention/dsa_backend.py').read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekSparseAttnBackend'))
        apply = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_cuda_graph_metadata'))
        block = next((n for n in apply.body if isinstance(n, ast.If) and 'self.dsa_decode_impl' in ast.unparse(n.test) and any((isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and (call.func.attr == 'copy_') for call in ast.walk(n)))))
        owner.dsa_decode_impl = 'flashmla_kv'
        schedules = []
        for size in [1, 2, 4, 8]:
            lengths = torch.full((size,), 2048, dtype=torch.int32)
            wrapped = owner._compute_flashmla_metadata(cache_seqlens=lengths, seq_len_q=1)
            scheduler = wrapped.flashmla_metadata
            scheduler.have_initialized = True
            scheduler.tile_scheduler_metadata = torch.ones(2, 4, dtype=torch.int32)
            scheduler.num_splits = torch.ones(1, dtype=torch.int32)
            pointer = scheduler.tile_scheduler_metadata.data_ptr()
            for offset in [0, 17, 1024]:
                lengths.fill_(2048 + offset)
                ns = {'self': owner, 'metadata': types.SimpleNamespace(flashmla_metadata=wrapped), 'dsa_cache_seqlens': lengths, 'seqlens_expanded_size': size}
                with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host item')), patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host cpu')), patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host list')):
                    exec(compile(ast.Module(body=[block], type_ignores=[]), 'actual_dsa_metadata_replay', 'exec'), ns)
                self.assertIs(wrapped.flashmla_metadata, scheduler)
                self.assertEqual(scheduler.tile_scheduler_metadata.data_ptr(), pointer)
                self.assertIsNone(wrapped.num_splits)
            schedules.append(scheduler)
        self.assertEqual(len({id(x) for x in schedules}), 4)

    def test_kda_zero_rows_are_handled_before_original_kernel(self):
        method = self.actual_method('srt/layers/attention/linear/kernels/kda_triton.py', 'TritonKDAKernel', 'packed_decode', {'torch': torch, 'fused_recurrent_kda_packed_decode': lambda **kw: kw['out'].zero_()})
        state = torch.ones(16, 2, 2, 2)
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host item')):
            output = COMPAT.around_kda_packed_decode(method, None, torch.zeros(0, 12), torch.zeros(0, 2, 2), torch.zeros(0, 1, 2), A_log=torch.zeros(2), dt_bias=torch.zeros(2, 2), scale=0.5, ssm_states=state, cache_indices=torch.zeros(0, dtype=torch.int64), num_v_heads=2, head_v_dim=2, lower_bound=-5.0)
        self.assertEqual(output.shape, (1, 0, 2, 2))
        torch.testing.assert_close(state, torch.ones_like(state))
        called = []
    
        def original(*a, **kw):
            called.append((a, kw))
            return 'unchanged'
        self.assertEqual(COMPAT.around_kda_packed_decode(original, 'owner', torch.ones(2, 12), 'a', 'b', num_v_heads=2, head_v_dim=2), 'unchanged')
        self.assertEqual(len(called), 1)

    def test_plugin_flashmla_helper_retains_fresh_hcu_metadata(self):
        module = load('hcu_flashmla_cpu_contract', REPO / 'hcu_sglang/layers/attention/current_dsa/flashmla_backend.py')
        before = module.DSAFlashMLAMetadata(object(), None)
        fresh = module.DSAFlashMLAMetadata(object(), None)
        self.assertIs(module.refresh_flashmla_metadata(before, fresh, slice(0, 2), is_hcu=True), fresh)
        self.assertFalse(module.can_fuse_flashmla_metadata(fresh))

    def test_dsa_capture_uses_variant_instead_of_host_sequence_lengths(self):
        mode = types.SimpleNamespace(is_extend_without_speculative=lambda: False, is_decode_or_idle=lambda: True)
        batch = types.SimpleNamespace(forward_mode=mode, seq_lens_cpu=torch.tensor([131072]))
        package('sglang.srt.model_executor')
        package('sglang.srt.model_executor.runner_utils')
        capture = package('sglang.srt.model_executor.runner_utils.capture_mode')
        method = self.actual_method('srt/layers/attention/dsa/dsa_indexer.py', 'Indexer', '_should_skip_logits_computation', {'dsa_shards_index_k': lambda: False, 'get_is_capture_mode': lambda: True, 'DSA_DENSE': 'dense'})
        for variant, expected in [(None, False), ('dense', True), ('sparse', False)]:
            capture.get_capture_attention_variant = lambda: variant
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host item')):
                self.assertEqual(method(types.SimpleNamespace(index_topk=2048), batch), expected)
    def test_deepep_graph_hooks_are_explicit_and_match_contract(self):
        package('sglang.srt.plugins')
        for name in ['sglang.srt.layers.attention','sglang.srt.layers.attention.linear','sglang.srt.layers.attention.linear.kernels']:
            package(name)
        registry=load('sglang.srt.plugins.hook_registry',SGLANG/'python/sglang/srt/plugins/hook_registry.py')
        kernel=package('sglang.srt.layers.attention.linear.kernels.kda_triton')
        dsa=package('sglang.srt.layers.attention.dsa_backend')
        method=self.actual_method('srt/layers/attention/linear/kernels/kda_triton.py','TritonKDAKernel','packed_decode',{'torch':torch})
        kernel.TritonKDAKernel=type('TritonKDAKernel',(),{'packed_decode':method})
        compute=self.actual_method('srt/layers/attention/dsa_backend.py','DeepseekSparseAttnBackend','_compute_flashmla_metadata',{'DSAFlashMLAMetadata':lambda **kw:types.SimpleNamespace(**kw)})
        dsa.DeepseekSparseAttnBackend=type('DeepseekSparseAttnBackend',(),{'_compute_flashmla_metadata':compute})
        for enabled in ['0','1']:
            registry.HookRegistry.reset()
            with patch.dict(os.environ,{'HCU_KIMI31_EP_TRANSPORT':'deepep','HCU_KIMI31_FULL_GRAPH_COMPAT':enabled}):COMPAT.register_hooks()
            if enabled=='0':
                self.assertEqual(len(registry.HookRegistry._hooks),0)
                continue
            registry.HookRegistry.apply_hooks()
            contract=json.loads((REPO/'tests/contracts/hcu_source_alignment_targets.json').read_text())
            self.assertEqual(registry.HookRegistry._patched,set(contract['optional_profiles']['kimi31_deepep_full_graph']['expected_hook_targets']))
            out=kernel.TritonKDAKernel().packed_decode(torch.zeros(0,12),torch.zeros(0,2,2),torch.zeros(0,1,2),num_v_heads=2,head_v_dim=2)
            self.assertEqual(out.shape,(1,0,2,2))
        registry.HookRegistry.reset()
if __name__=='__main__':unittest.main(verbosity=2)

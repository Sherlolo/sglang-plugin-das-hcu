import sys
import types
import unittest
from unittest.mock import patch

import torch

from hcu_sglang import kimi31_flashkda as bridge
from sglang.srt.layers.attention.linear.kernels.kda_flashkda import FlashKDAKernel
from sglang.srt.plugins.hook_registry import HookType


def inputs(lengths=(64, 64), slots=(1, 3)):
    heads = 3
    tokens = sum(lengths)
    q = torch.randn(1, tokens, heads, 128, dtype=torch.bfloat16)
    values = dict(q=q, k=q.clone(), v=q.clone(), g=q.clone(),
                  beta=torch.randn(1, tokens, heads, dtype=torch.bfloat16),
                  ssm_states=torch.randn(12, heads, 128, 128,
                                         dtype=torch.bfloat16)[::2],
                  cache_indices=torch.tensor(slots, dtype=torch.int32),
                  query_start_loc=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                                               dtype=torch.int32),
                  A_log=torch.zeros(heads), dt_bias=torch.zeros(heads * 128),
                  lower_bound=-5.0, beta_is_raw=True,
                  extend_seq_lens_cpu=list(lengths))
    return values


class FlashKDABridgeTests(unittest.TestCase):
    def run_long_case(self, lengths, cp_flag):
        args = inputs(lengths=lengths)
        state = args['ssm_states'].clone()
        observed = []
        fake = types.ModuleType('flash_kda')

        def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                *, initial_state, final_state, cu_seqlens, enable_cp, _dispatch_info):
            observed.append((tuple(q.shape), tuple(beta.shape), enable_cp,
                             None if cu_seqlens is None else cu_seqlens.tolist()))
            torch.testing.assert_close(initial_state, state[args['cache_indices']])
            self.assertTrue(initial_state.is_contiguous())
            self.assertEqual(A_log.dtype, torch.float32)
            self.assertEqual(dt_bias.shape, (3, 128))
            self.assertEqual(scale, 128 ** -0.5)
            final_state.copy_(initial_state + 1)
            out.fill_(0.5)
            _dispatch_info.update(route='test', reason='test')

        fake.fwd = fwd
        with patch.dict('os.environ', {'HCU_KIMI31_FLASHKDA_LONG_PREFILL': '1',
                                       'HCU_KIMI31_FLASHKDA_FIXED_CP': cp_flag}):
            with patch.dict(sys.modules, {'flash_kda': fake}):
                result = bridge._hcu_extend(
                    FlashKDAKernel(), args['q'], args['k'], args['v'],
                    args['g'], args['beta'],
                    **{key: value for key, value in args.items()
                       if key not in ('q', 'k', 'v', 'g', 'beta')},
                )
        self.assertEqual(result.shape, args['v'].shape)
        self.assertTrue(torch.all(result == 0.5))
        torch.testing.assert_close(args['ssm_states'][args['cache_indices']],
                                   state[args['cache_indices']] + 1)
        torch.testing.assert_close(args['ssm_states'][0], state[0])
        self.assertEqual(len(observed), 1)
        return observed[0]

    def test_long_equal_sequences_use_fixed_layout_and_scatter_state(self):
        self.assertEqual(self.run_long_case((2112, 2112), '1'),
                         ((2, 2112, 3, 128), (2, 2112, 3), True, None))

    def test_long_mixed_sequences_preserve_varlen_offsets(self):
        self.assertEqual(self.run_long_case((2112, 64), '0'),
                         ((1, 2176, 3, 128), (1, 2176, 3), False, [0, 2112, 2176]))

    def test_long_sequence_guard_remains_opt_in(self):
        args = inputs(lengths=(2112, 64))
        common = {key: value for key, value in args.items()
                  if key not in ('q', 'k', 'v', 'g', 'beta')}
        with patch.dict('os.environ', {'HCU_KIMI31_FLASHKDA_LONG_PREFILL': '0'}):
            reason = bridge._fallback_reason(
                args['q'], args['k'], args['v'], args['g'], args['beta'],
                **common, is_spec_decode=False, return_intermediate_states=False)
        self.assertEqual(reason, 'sequence_lengths')

    def test_fused_path_returns_tensor_and_scatters_strided_bf16_state(self):
        args = inputs()
        original_state = args["ssm_states"].clone()
        observed = []
        fake = types.ModuleType("flash_kda")

        def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                *, initial_state, final_state, cu_seqlens):
            observed.append((initial_state.clone(), initial_state.is_contiguous(),
                             tuple(cu_seqlens.tolist()), beta.dtype))
            final_state.copy_(initial_state + 1)
            out.fill_(0.25)

        fake.fwd = fwd
        with patch.dict(sys.modules, {"flash_kda": fake}):
            result = bridge._hcu_extend(
                FlashKDAKernel(), args["q"], args["k"], args["v"],
                args["g"], args["beta"],
                **{key: value for key, value in args.items()
                   if key not in ("q", "k", "v", "g", "beta")},
            )
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, args["v"].shape)
        self.assertTrue(torch.all(result == 0.25))
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0][1])
        self.assertEqual(observed[0][2], (0, 64, 128))
        self.assertEqual(observed[0][3], torch.bfloat16)
        torch.testing.assert_close(args["ssm_states"][args["cache_indices"].long()],
                                   original_state[args["cache_indices"].long()] + 1)
        torch.testing.assert_close(args["ssm_states"][0], original_state[0])

    def test_fallback_contract_rejects_padding_duplicates_and_tracking(self):
        args = inputs()
        common = {key: value for key, value in args.items()
                  if key not in ("q", "k", "v", "g", "beta")}
        self.assertIsNone(bridge._fallback_reason(args["q"], args["k"],
                                                   args["v"], args["g"],
                                                   args["beta"], **common,
                                                   is_spec_decode=False,
                                                   return_intermediate_states=False))
        for slots in ((-1, 3), (1, 1), (1, 6)):
            bad = {**common, "cache_indices": torch.tensor(slots)}
            self.assertEqual(bridge._fallback_reason(
                args["q"], args["k"], args["v"], args["g"], args["beta"],
                **bad, is_spec_decode=False, return_intermediate_states=False),
                "state_indices")
        self.assertEqual(bridge._fallback_reason(
            args["q"], args["k"], args["v"], args["g"], args["beta"],
            **common, is_spec_decode=False, return_intermediate_states=True),
            "state_snapshot_or_speculation")

    def test_hook_targets_real_flashkda_method(self):
        registrations = []
        with patch("sglang.srt.plugins.hook_registry.HookRegistry.register",
                   side_effect=lambda *args: registrations.append(args)):
            bridge.register_hooks()
        self.assertEqual(registrations, [(
            "sglang.srt.layers.attention.linear.kernels.kda_flashkda.FlashKDAKernel.extend",
            bridge.around_extend, HookType.AROUND,
        )])

    def test_tracked_prefill_uses_triton_kernel_return_contract(self):
        args = inputs()
        fake_module = types.ModuleType(
            "sglang.srt.layers.attention.linear.kernels.kda_triton"
        )
        calls = []

        class TritonKDAKernel:
            def extend(self, *positional, **keyword):
                calls.append((positional, keyword))
                return ("triton_output", "track_state")

        fake_module.TritonKDAKernel = TritonKDAKernel
        with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            result = bridge._hcu_extend(
                FlashKDAKernel(), args["q"], args["k"], args["v"],
                args["g"], args["beta"],
                **{key: value for key, value in args.items()
                   if key not in ("q", "k", "v", "g", "beta")},
                return_intermediate_states=True,
            )
        self.assertEqual(result, ("triton_output", "track_state"))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["return_intermediate_states"])

    def test_empty_prefill_does_not_touch_state_or_kernel(self):
        args = inputs(lengths=(), slots=())
        args["q"] = args["q"][:, :0]
        state = args["ssm_states"].clone()
        result = bridge._hcu_extend(
            FlashKDAKernel(), args["q"], args["k"], args["v"],
            args["g"], args["beta"],
            **{key: value for key, value in args.items()
               if key not in ("q", "k", "v", "g", "beta")},
        )
        self.assertEqual(result.shape, args["v"].shape)
        torch.testing.assert_close(args["ssm_states"], state)


class ForwardMetadataCacheTests(unittest.TestCase):
    def batch(self, extend=True):
        return types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_extend_without_speculative=lambda: extend))

    def run_forward(self, original, batch=None, draft=False, *args, **kwargs):
        with patch.object(torch.version, 'hip', '7.2'):
            return bridge.around_model_forward(
                original, types.SimpleNamespace(is_draft_worker=draft),
                batch or self.batch(), *args, **kwargs)

    def test_forward_reuses_readonly_metadata_and_forwards_arguments(self):
        metadata = torch.tensor([0, 64], dtype=torch.int32)
        copies = []
        original_cpu = torch.Tensor.cpu

        def cpu(tensor, *args, **kwargs):
            copies.append(tensor.numel())
            return original_cpu(tensor, *args, **kwargs)

        sentinel = object()

        def forward(owner, batch, *args, **kwargs):
            self.assertEqual(args, (None,))
            self.assertIs(kwargs['pp_proxy_tensors'], sentinel)
            first = bridge._metadata_list(metadata)
            first[0] = 99
            self.assertEqual(bridge._metadata_list(metadata), [0, 64])
            return 'result'

        with patch.object(torch.Tensor, 'cpu', cpu):
            result = self.run_forward(forward, None, False, None,
                                      pp_proxy_tensors=sentinel)
        self.assertEqual(result, 'result')
        self.assertEqual(copies, [2])
        self.assertIsNone(bridge._forward_metadata_cache.get())

    def test_inference_metadata_refreshes_on_next_real_forward(self):
        with torch.inference_mode():
            metadata = torch.tensor([0, 64], dtype=torch.int32)
            def forward(owner, batch):
                self.assertEqual(bridge._metadata_list(metadata),
                                 bridge._metadata_list(metadata))
                return bridge._metadata_list(metadata)
            self.assertEqual(self.run_forward(forward), [0, 64])
            metadata[1] = 128
            self.assertEqual(self.run_forward(forward), [0, 128])
        self.assertIsNone(bridge._forward_metadata_cache.get())

    def test_versioned_inplace_mutation_invalidates_within_forward(self):
        metadata = torch.tensor([0, 64], dtype=torch.int32)
        def forward(owner, batch):
            self.assertEqual(bridge._metadata_list(metadata), [0, 64])
            metadata[1] = 128
            self.assertEqual(bridge._metadata_list(metadata), [0, 128])
        self.run_forward(forward)

    def test_decode_and_draft_keep_uncached_original_path(self):
        metadata = torch.tensor([0, 64], dtype=torch.int32)
        def forward(owner, batch):
            self.assertIsNone(bridge._forward_metadata_cache.get())
            return bridge._metadata_list(metadata)
        self.assertEqual(self.run_forward(forward, self.batch(False)), [0, 64])
        self.assertEqual(self.run_forward(forward, draft=True), [0, 64])

    def test_nested_scope_and_exception_restore_outer_scope(self):
        metadata = torch.tensor([0, 64], dtype=torch.int32)
        def inner(owner, batch):
            self.assertEqual(bridge._metadata_list(metadata), [0, 64])
            raise RuntimeError('inner failure')
        def outer(owner, batch):
            self.assertEqual(bridge._metadata_list(metadata), [0, 64])
            cache = bridge._forward_metadata_cache.get()
            with self.assertRaisesRegex(RuntimeError, 'inner failure'):
                self.run_forward(inner)
            self.assertIs(bridge._forward_metadata_cache.get(), cache)
            self.assertEqual(bridge._metadata_list(metadata), [0, 64])
            raise RuntimeError('outer failure')
        with self.assertRaisesRegex(RuntimeError, 'outer failure'):
            self.run_forward(outer)
        self.assertIsNone(bridge._forward_metadata_cache.get())

    def test_scope_hook_is_explicitly_opt_in(self):
        target = 'sglang.srt.model_executor.model_runner.ModelRunner.forward'
        from sglang.srt.plugins.hook_registry import HookRegistry
        for value, expected in [('0', False), ('1', True)]:
            with patch.dict('os.environ', {'HCU_KIMI31_CPU_METADATA_CACHE': value}):
                with patch.object(HookRegistry, 'register') as register:
                    bridge.register_hooks()
            self.assertEqual(target in [c.args[0] for c in register.call_args_list],
                             expected)


if __name__ == "__main__":
    unittest.main()

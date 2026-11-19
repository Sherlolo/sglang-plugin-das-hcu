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


if __name__ == "__main__":
    unittest.main()

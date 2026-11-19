from contextlib import contextmanager, nullcontext
import os
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest

from hcu_sglang.kimi31_dspark import around_draft_context, register_dp_deepep


class Parallel(S):
    @contextmanager
    def override(self, **values):
        before = {key: getattr(self, key) for key in values}
        self.__dict__.update(values)
        try:
            yield
        finally:
            self.__dict__.update(before)


class DSparkBackendPolicyTests(unittest.TestCase):
    def test_native_deepep_extends_only_the_kimi_policy(self):
        from sglang.srt.arg_groups import model_override_base as policy

        architecture = "KimiK31ForConditionalGeneration"
        with patch.dict(policy._DSPARK_DP_A2A_BACKENDS, clear=True):
            default = policy.dspark_dp_a2a_backends(architecture)
            with patch.dict(os.environ, {
                "HCU_KIMI31_DEEPEP": "1",
                "HCU_KIMI31_EP_TRANSPORT": "nccl",
            }):
                register_dp_deepep()
                self.assertEqual(policy.dspark_dp_a2a_backends(architecture), default)
            with patch.dict(os.environ, {
                "HCU_KIMI31_DEEPEP": "1",
                "HCU_KIMI31_EP_TRANSPORT": "deepep",
            }):
                register_dp_deepep()
                register_dp_deepep()
            self.assertEqual(policy.dspark_dp_a2a_backends(architecture), (*default, "deepep"))
            self.assertEqual(policy.dspark_dp_a2a_backends("OtherModel"), default)


class DSparkDraftContextTests(unittest.TestCase):
    def test_dense_scope_restores_target_state_on_error(self):
        parallel = Parallel(disable_attn_tp_gather=False)
        calls = []

        @contextmanager
        def original(worker):
            calls.append('enter')
            try:
                yield
            finally:
                calls.append('exit')

        with patch('sglang.srt.runtime_context.get_parallel', return_value=parallel):
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                with around_draft_context(original, S(_draft_is_moe=False)):
                    self.assertTrue(parallel.disable_attn_tp_gather)
                    raise RuntimeError('sentinel')
        self.assertFalse(parallel.disable_attn_tp_gather)
        self.assertEqual(calls, ['enter', 'exit'])

    def test_moe_draft_keeps_target_gather_policy(self):
        parallel = Parallel(disable_attn_tp_gather=False)
        with patch('sglang.srt.runtime_context.get_parallel', return_value=parallel):
            with around_draft_context(lambda worker: nullcontext(), S(_draft_is_moe=True)):
                self.assertFalse(parallel.disable_attn_tp_gather)

    def test_real_graph_sizing_accepts_seven_token_dense_draft_at_tp16(self):
        import sglang.srt.layers.dp_attention as dp
        import sglang.srt.model_executor.runner.base_cuda_graph_runner as graph

        parallel = Parallel(disable_attn_tp_gather=False, attn_tp_size=16,
                            attn_cp_size=1, attn_dp_enabled=False,
                            moe_dense_tp_size=None)
        execution = S(graph=S(cuda_graph_config=S(decode=S(bs=[1, 2, 4, 8]))),
                      overlap=S(enable_two_batch_overlap=False))
        runner = S(req_to_token_pool=S(size=8), server_args=S())
        with (patch('sglang.srt.runtime_context.get_parallel', return_value=parallel),
              patch.object(dp, 'get_parallel', return_value=parallel),
              patch.object(dp, 'get_exec', return_value=execution),
              patch.object(graph, 'get_exec', return_value=execution),
              patch.object(graph, 'get_flags', return_value=S(capture=S(enable_torch_compile=False))),
              patch.object(graph, 'can_defer_cp_decode_attn_tp', return_value=False),
              patch.object(graph.DllmConfig, 'from_server_args', return_value=None),
              patch('sglang.srt.layers.moe.utils.get_moe_a2a_backend', return_value=S(is_none=lambda: False))):
            self.assertTrue(dp.require_attn_tp_gather())
            with self.assertRaises(AssertionError):
                graph.get_batch_sizes_to_capture(runner, 7, gathered_buffer_required=dp.require_attn_tp_gather())
            with around_draft_context(lambda worker: nullcontext(), S(_draft_is_moe=False)):
                self.assertFalse(dp.require_attn_tp_gather())
                sizes, compile_sizes = graph.get_batch_sizes_to_capture(runner, 7, gathered_buffer_required=dp.require_attn_tp_gather())
                self.assertEqual(sizes, [1, 2, 4, 8])
                self.assertEqual(compile_sizes, [])
            self.assertTrue(dp.require_attn_tp_gather())


if __name__ == '__main__':
    unittest.main()

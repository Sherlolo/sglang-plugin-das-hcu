import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kimi31_communication_test', ROOT / 'hcu_sglang/kimi31_communication.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class CommunicationTests(unittest.TestCase):
    def test_day0_reaches_opt_in_registration(self):
        tree = ast.parse((ROOT / 'hcu_sglang/kimi31_day0.py').read_text())
        registration = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'register_hooks')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and isinstance(n.func.value, ast.Name)
                            and n.func.value.id == 'kimi31_communication' and n.func.attr == 'register_hooks'
                            for n in ast.walk(registration)))
        registry = NS(register=Mock())
        environment = NS(register_envs=Mock())
        with patch.dict(os.environ, {'SGLANG_USE_AITER_AR': '1'}), patch.dict(sys.modules, {
                'hcu_sglang.environ': environment,
                'sglang.srt.plugins.hook_registry': NS(HookRegistry=registry, HookType=NS(AROUND=object()))}):
            C.register_hooks()
        self.assertEqual(registry.register.call_count, 3)
        environment.register_envs.assert_called_once_with()

    def test_registered_communicator_receives_live_capture_inputs(self):
        communicator = NS(disabled=False, _IS_CAPTURING=True,
                          should_custom_ag=lambda x: True, should_custom_ar=lambda x: True,
                          all_gather_unreg=Mock(), reduce_scatter=Mock())
        group = NS(ca_comm=communicator)
        input = NS(numel=lambda: 4)
        output = object()
        original = Mock()
        with patch.dict(os.environ, {'SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM': '1'}), patch.dict(sys.modules, {
                'hcu_sglang.utils.detection': NS(is_hcu=lambda: True),
                'torch': NS(cuda=NS(is_current_stream_capturing=lambda: True))}):
            self.assertIs(C.around_reduce_scatter(original, group, output, input), output)
            C.around_all_gather(original, group, output, input)
        communicator.reduce_scatter.assert_called_once_with(input, output, registered=False)
        communicator.all_gather_unreg.assert_called_once_with(input, out=output)
        original.assert_not_called()

    def test_explicit_plugin_aiter_cannot_override_adapter_veto(self):
        provider = Mock()
        original = Mock()
        for disabled, policy in ((True, '1'), (False, 'refused(old-aiter)')):
            adapter = NS(hcu_env=NS(resolve=lambda key: (policy, 'test')),
                         kill_switch=NS(op_disabled=lambda name: disabled))
            with patch.dict(sys.modules, {
                    'hcu_sglang.config': NS(resolved_custom_all_reduce_backend=lambda: 'aiter'),
                    'hcu_sglang.utils.detection': NS(is_hcu=lambda: True),
                    'hcu_sglang.distributed_aiter': NS(dispatch_custom_allreduce=provider),
                    'hcu_adapter': adapter}):
                with self.assertRaisesRegex(RuntimeError, 'adapter policy disabled'):
                    C.around_dispatch(original, object(), object())
        provider.assert_not_called()
        original.assert_not_called()

    def test_shared_storage_reduce_scatter_keeps_original(self):
        original = Mock(return_value='baseline')
        storage = NS(data_ptr=lambda: 1234)
        input = NS(untyped_storage=lambda: storage)
        output = NS(untyped_storage=lambda: storage)
        communicator = Mock()
        with patch.object(C, '_communicator', return_value=communicator):
            self.assertEqual(C.around_reduce_scatter(original, NS(), output, input), 'baseline')
        original.assert_called_once()
        communicator.reduce_scatter.assert_not_called()

    def test_all_gather_flattens_native_output_contract(self):
        original = Mock()
        communicator = Mock()
        input = NS(view=Mock(return_value='flat_in'), numel=lambda: 4, is_contiguous=lambda: True, dtype='bf16', device='gpu')
        output = NS(view=Mock(return_value='flat_out'), numel=lambda: 32, is_contiguous=lambda: True, dtype='bf16', device='gpu')
        with patch.object(C, '_communicator', return_value=communicator):
            C.around_all_gather(original, NS(world_size=8), output, input)
        communicator.all_gather_unreg.assert_called_once_with('flat_in', out='flat_out')
        original.assert_not_called()

    def test_unavailable_communicator_keeps_previous_wrapper(self):
        original = Mock(return_value='baseline')
        with patch.dict(os.environ, {'SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM': '1'}), patch.dict(sys.modules, {
                'hcu_sglang.utils.detection': NS(is_hcu=lambda: True)}):
            self.assertEqual(C.around_reduce_scatter(original, NS(ca_comm=None), None, None), 'baseline')
        original.assert_called_once()


if __name__ == '__main__':
    unittest.main()

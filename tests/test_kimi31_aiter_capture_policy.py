import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


class AiterCapturePolicyTests(unittest.TestCase):
    def test_hcu_capture_allgather_without_fork_global(self):
        path = Path(__file__).resolve().parents[1] / 'hcu_sglang/distributed_methods.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_all_gather_into_tensor')
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        communicator = NS(disabled=False, _IS_CAPTURING=True,
                          should_custom_ag=lambda x: True,
                          enable_register_for_capturing=True,
                          all_gather_unreg=Mock(), all_gather_reg=Mock())
        group = NS(ca_comm=communicator, _has_aiter_custom_all_gather=lambda: True)
        source = NS(is_contiguous=lambda: True, dtype='bf16')
        output = NS(is_contiguous=lambda: True)
        namespace = {
            '_ATTN_TP_USE_AITER_CUSTOM_COMM': False,
            '_is_hcu': True,
            'is_hip': lambda: True,
            'envs': NS(SGLANG_USE_AITER_AG=NS(get=lambda: True),
                       SGLANG_MEMORY_SAVER_CUDA_GRAPH=NS(get=lambda: False),
                       SGLANG_AITER_CAPTURE_COPY_IN=NS(get=lambda: False)),
            'torch': NS(cuda=NS(is_current_stream_capturing=lambda: True),
                        float32='fp32', float16='fp16', bfloat16='bf16'),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        namespace['_all_gather_into_tensor'](group, output, source)
        communicator.all_gather_unreg.assert_called_once_with(source, out=output, dim=0)
        communicator.all_gather_reg.assert_not_called()


if __name__ == '__main__':
    unittest.main()

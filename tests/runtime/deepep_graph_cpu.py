from __future__ import annotations
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor.runner_utils import deepep_adapter as A
BASE_CLASS=[A.DeepEPCudaGraphRunnerAdapter]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class DeepEPGraphCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def buffer(self):
        class Buffer:
            layout=None
            events=[]
            @classmethod
            def get_low_latency_layout(cls):return cls.layout
            @classmethod
            def set_low_latency_layout(cls,*layout):cls.layout=layout;cls.events.append(('layout',layout))
            @classmethod
            def set_dispatch_mode(cls,mode):cls.events.append(('mode',mode))
        return Buffer

    def test_native_class_alias_remains_class_with_compatible_owned_instances(self):
        from hcu_sglang import deepep_graph as O
        self.assertIs(A.DeepEPCudaGraphRunnerAdapter,O.DeepEPCudaGraphRunnerAdapter)
        self.assertTrue(issubclass(A.DeepEPCudaGraphRunnerAdapter,BASE_CLASS[0]))
        self.assertIsInstance(A.DeepEPCudaGraphRunnerAdapter(),BASE_CLASS[0])

    def test_replay_restores_captured_target_layout_before_dispatch_mode(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.AUTO):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(False)
            B.layout=(128,7168,264);obj.capture_end()
            B.layout=(128,7168,256);B.events.clear();obj.replay()
        self.assertEqual(B.events,[('layout',(128,7168,264)),('mode',DeepEPMode.LOW_LATENCY)])

    def test_normal_or_non_deepep_do_not_restore_absent_layout(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.NORMAL):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(True);obj.capture_end();B.events.clear();obj.replay()
            self.assertEqual(B.events,[('mode',DeepEPMode.NORMAL)])
        B.events.clear()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.NONE):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(False);obj.capture_end();obj.replay()
        self.assertEqual(B.events,[])

    def test_capture_delegates_current_signature_and_records_completed_layout(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer();seen=[];result=object()
        def original(runner,size,forward,variant_label=None,attention_variant=None):
            seen.append((size,forward,variant_label,attention_variant))
            runner.deepep_adapter.capture(False)
            B.layout=(128,7168,264)
            return result
        with patch.object(O,'is_hcu',return_value=True),patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.AUTO):
            runner=S(deepep_adapter=A.DeepEPCudaGraphRunnerAdapter())
            self.assertIs(O.around_capture(original,runner,4,'forward','variant','attention'),result)
        self.assertEqual(seen,[(4,'forward','variant','attention')])
        self.assertEqual(runner.deepep_adapter._captured_low_latency_layout,(128,7168,264))

    def test_failed_capture_does_not_record_partial_layout(self):
        from hcu_sglang import deepep_graph as O
        ended=[];runner=S(deepep_adapter=S(capture_end=lambda:ended.append(True)))
        def original(*args,**kwargs):raise RuntimeError('capture failed')
        with patch.object(O,'is_hcu',return_value=True),self.assertRaisesRegex(RuntimeError,'capture failed'):
            O.around_capture(original,runner,1,None)
        self.assertEqual(ended,[])

    def test_non_hcu_capture_keeps_effective_original(self):
        from hcu_sglang import deepep_graph as O
        token=object();seen=[]
        with patch.object(O,'is_hcu',return_value=False):
            self.assertIs(O.around_capture(lambda *a,**k:seen.append((a,k)) or token,token,4,variant_label='x'),token)
        self.assertEqual(seen,[((token,4),{'variant_label':'x'})])


if __name__=='__main__':unittest.main()

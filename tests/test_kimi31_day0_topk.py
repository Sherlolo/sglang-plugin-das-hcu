import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('day0_topk',Path(__file__).parents[1]/'hcu_sglang/kimi31_topk.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class Day0TopkTests(unittest.TestCase):
    def test_paged_real_upstream_enum_routes_vendor(self):
        fused=Mock(return_value=object())
        backend=SimpleNamespace(is_sgl_kernel=lambda:True)
        pages,score,lengths,cu=object(),object(),object(),object()
        detection=SimpleNamespace(is_hcu=lambda:True)
        with patch.dict(os.environ,{'SGLANG_DSA_FUSE_TOPK':'1'}),patch.dict(sys.modules,{
            'hcu_sglang.utils.detection':detection,
            'lightop':SimpleNamespace(fast_topk_transform_fused=fused,fast_topk_transform_ragged_fused=Mock())}):
            out=M.around_transform(Mock(side_effect=AssertionError('upstream CUDA called')),backend,score,lengths,2048,SimpleNamespace(name='PAGED'),SimpleNamespace(page_table_1=pages),cu_seqlens_q_topk=cu)
        self.assertIs(out,fused.return_value)
        fused.assert_called_once_with(score=score,lengths=lengths,page_table_size_1=pages,cu_seqlens_q=cu,topk=2048,row_starts=None)

    def test_disabled_retains_existing_wrapper_and_new_kwargs(self):
        original=Mock(return_value=object())
        with patch.dict(os.environ,{'SGLANG_DSA_FUSE_TOPK':'0'}),patch.dict(sys.modules,{'hcu_sglang.utils.detection':SimpleNamespace(is_hcu=lambda:True)}):
            out=M.around_transform(original,SimpleNamespace(is_sgl_kernel=lambda:True),'s','l',2048,'method','meta',packed_extend_topk=True)
        self.assertIs(out,original.return_value)
        self.assertTrue(original.call_args.kwargs['packed_extend_topk'])

if __name__=='__main__':unittest.main()

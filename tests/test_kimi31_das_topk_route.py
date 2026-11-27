import ast
import enum
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class DasTopkRouteTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "hcu_sglang/layers/attention/current_dsa/dsa_topk_backend.py"
        tree = ast.parse(source.read_text())
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        nodes += [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("DSATopKBackend", "TopkTransformMethod")]
        envs = SimpleNamespace(
            SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda: True),
            SGLANG_OPT_USE_TOPK_V2=SimpleNamespace(get=lambda: False),
        )
        namespace = dict(Enum=enum.Enum, IntEnum=enum.IntEnum, auto=enum.auto,
                         envs=envs, _is_hcu=True, _is_hip=True)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
        self.backend = namespace["DSATopKBackend"].SGL_KERNEL
        self.methods = namespace["TopkTransformMethod"]

    def test_paged_fused_uses_lightop_and_forwards_metadata(self):
        fused, ragged = Mock(return_value=object()), Mock()
        lightop = SimpleNamespace(fast_topk_transform_fused=fused,
                                  fast_topk_transform_ragged_fused=ragged)
        score, lengths, page, cu = object(), object(), object(), object()
        metadata = SimpleNamespace(page_table_1=page, token_to_batch_idx=None)
        with patch.dict(sys.modules, {"lightop": lightop}):
            result = self.backend.topk_transform(score, lengths, 2048,
                self.methods.PAGED, metadata, cu_seqlens_q_topk=cu)
        self.assertIs(result, fused.return_value)
        fused.assert_called_once_with(score=score, lengths=lengths,
            page_table_size_1=page, cu_seqlens_q=cu, topk=2048, row_starts=None)
        ragged.assert_not_called()

    def test_ragged_fused_requires_and_forwards_offsets(self):
        fused, ragged = Mock(), Mock(return_value=object())
        lightop = SimpleNamespace(fast_topk_transform_fused=fused,
                                  fast_topk_transform_ragged_fused=ragged)
        score, lengths, offsets = object(), object(), object()
        metadata = SimpleNamespace(page_table_1=object(), token_to_batch_idx=None)
        with patch.dict(sys.modules, {"lightop": lightop}):
            result = self.backend.topk_transform(score, lengths, 2048,
                self.methods.RAGGED, metadata, topk_indices_offset=offsets)
            with self.assertRaisesRegex(RuntimeError, "topk_indices_offset"):
                self.backend.topk_transform(score, lengths, 2048,
                    self.methods.RAGGED, metadata)
        self.assertIs(result, ragged.return_value)
        ragged.assert_called_once_with(score=score, lengths=lengths,
            topk_indices_offset=offsets, topk=2048, row_starts=None)
        fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()

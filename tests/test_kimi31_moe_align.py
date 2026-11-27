import importlib.util
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PATH = Path(__file__).resolve().parents[1] / "hcu_sglang/kimi31_moe_align.py"
SPEC = importlib.util.spec_from_file_location("kimi31_moe_align_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Kimi31MoeAlignTests(unittest.TestCase):
    def test_default_preserves_effective_baseline(self):
        original = Mock()
        args = (SimpleNamespace(numel=lambda: 32), 449, 64, object(), object(), object())
        with patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "0"}):
            MODULE.align_hcu_tokens(original, *args)
        original.assert_called_once_with(*args)

    def test_preallocated_output_abi_and_expert_padding_are_preserved(self):
        original = Mock()
        align = Mock()
        args = (SimpleNamespace(numel=lambda: 1024), 449, 64, object(), object(), object())
        lightop = SimpleNamespace(moe=SimpleNamespace(moe_align_block_size_out=align))
        with patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}), patch.dict(sys.modules, {"lightop": lightop}):
            MODULE.align_hcu_tokens(original, *args)
        align.assert_called_once_with(*args, None, None, None, False, True)
        original.assert_not_called()

    def test_oversized_and_empty_do_not_import_vendor(self):
        for count in (0, 1025, 4096):
            original = Mock()
            args = (SimpleNamespace(numel=lambda: count), 449, 64, object(), object(), object())
            with patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}), patch.dict(sys.modules, {"lightop": None}):
                MODULE.align_hcu_tokens(original, *args)
            original.assert_called_once_with(*args)

    def test_vendor_failure_is_not_replayed(self):
        original = Mock()
        align = Mock(side_effect=RuntimeError("kernel failed"))
        lightop = SimpleNamespace(moe=SimpleNamespace(moe_align_block_size_out=align))
        args = (SimpleNamespace(numel=lambda: 32), 449, 64, object(), object(), object())
        with patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}), patch.dict(sys.modules, {"lightop": lightop}):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                MODULE.align_hcu_tokens(original, *args)
        original.assert_not_called()

    def test_unverified_expert_and_block_shapes_preserve_baseline(self):
        for experts, block in ((17, 64), (449, 32), (449, 128)):
            original = Mock()
            args = (SimpleNamespace(numel=lambda: 128), experts, block,
                    object(), object(), object())
            with patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}), patch.dict(sys.modules, {"lightop": None}):
                MODULE.align_hcu_tokens(original, *args)
            original.assert_called_once_with(*args)

    def test_aiter_moe_c_consumer_uses_lightop_and_applies_expert_map(self):
        source = PATH.parent / "layers/quantization/slimquant_w4a8_marlin.py"
        tree = ast.parse(source.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_patch_moe_c_align_non_cub_to_triton")
        outputs = [Mock(), Mock(), Mock()]
        torch = SimpleNamespace(int32=object(), empty=Mock(side_effect=outputs))
        original = Mock()
        fmc = SimpleNamespace(moe_align_block_size=original,
                              moe_align_block_size_triton=Mock())
        aiter = SimpleNamespace(fused_moe_c=fmc)
        triton = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b)
        namespace = dict(_moe_c_align_patched=False, _is_hcu=True,
                         torch=torch, logger=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(source), "exec"), namespace)
        ids = SimpleNamespace(numel=lambda: 1024, device=object())
        expert_map = Mock()
        expert_map.__getitem__ = Mock(return_value=object())
        align = Mock()
        modules = {"aiter": aiter, "aiter.fused_moe_c": fmc, "triton": triton,
                   "hcu_sglang.kimi31_moe_align": MODULE,
                   "lightop": SimpleNamespace(moe=SimpleNamespace(moe_align_block_size_out=align))}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}):
            namespace["_patch_moe_c_align_non_cub_to_triton"]()
            result = fmc.moe_align_block_size(ids, 64, 449, expert_map)
        align.assert_called_once_with(ids, 449, 64, *outputs, None, None, None, False, True)
        outputs[0].fill_.assert_called_once_with(1024)
        outputs[1].zero_.assert_called_once_with()
        expert_map.__getitem__.assert_called_once_with(outputs[1])
        self.assertIs(result[1], expert_map.__getitem__.return_value)
        original.assert_not_called()
        fmc.moe_align_block_size_triton.assert_not_called()

    def test_boltop_consumer_passes_initialized_outputs_and_expert_map(self):
        sorted_ids, expert_ids, padded = object(), object(), object()
        torch = SimpleNamespace(int32=object(), full=Mock(return_value=sorted_ids),
            zeros=Mock(return_value=expert_ids), empty=Mock(return_value=padded))
        ids = SimpleNamespace(numel=lambda: 1024, device=object())
        expert_map = Mock()
        expert_map.__getitem__ = Mock(return_value=object())
        align, original = Mock(), Mock()
        modules = {"torch": torch,
            "lightop": SimpleNamespace(moe=SimpleNamespace(moe_align_block_size_out=align))}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HCU_KIMI31_LIGHTOP_MOE_ALIGN": "1"}):
            result = MODULE.around_boltop_align(original, ids, 64, 449, expert_map)
        align.assert_called_once_with(ids, 449, 64, sorted_ids, expert_ids, padded,
            None, None, None, False, True)
        torch.full.assert_called_once_with((((1024+449*63+3)//4)*4,), 1024,
            dtype=torch.int32, device=ids.device)
        expert_map.__getitem__.assert_called_once_with(expert_ids)
        self.assertIs(result[1], expert_map.__getitem__.return_value)
        original.assert_not_called()


if __name__ == "__main__":
    unittest.main()

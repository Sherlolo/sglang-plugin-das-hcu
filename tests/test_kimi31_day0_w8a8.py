# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, sentinel


_HAVE_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "sglang", "transformers", "lightop")
)
_DRAFT_ROOT = Path(
    os.environ.get(
        "KIMI31_DSPARK_W8A8_ROOT",
        "/public/home/yangyn1/SothisAI/model/Console/Kimi-K3.1-Pro-DSpark-w8a8",
    )
)
_DRAFT_CONFIG = _DRAFT_ROOT / "main/Kimi-K3.1-Pro-DSpark-w8a8/config.json"


@unittest.skipUnless(_HAVE_RUNTIME, "requires torch, sglang, transformers and lightop")
class Kimi31Day0W8A8Tests(unittest.TestCase):
    def test_day0_registers_hcu_dense_int8_draft_dispatch(self):
        from hcu_sglang import kimi31_day0
        from hcu_sglang.layers.quantization import compressed_int8
        from sglang.srt.plugins.hook_registry import HookType

        hooks = {}

        def record(target, callback, hook_type):
            hooks[target] = (callback, hook_type)

        with (
            patch.dict(os.environ, {"HCU_KIMI31_DEEPEP": "0"}),
            patch.object(kimi31_day0.HookRegistry, "register", side_effect=record),
            patch.object(kimi31_day0.jit_overlay_loader, "register_hooks"),
            patch.object(kimi31_day0.configs, "register_hooks"),
            patch.object(kimi31_day0.server_args, "register"),
            patch.object(kimi31_day0.registry, "register_hooks"),
            patch.object(kimi31_day0.registry, "install_quantization_method"),
            patch.object(kimi31_day0, "_flashmla_operation", return_value=Mock()),
            patch.object(compressed_int8, "is_hcu", return_value=True),
        ):
            kimi31_day0.register_hooks()

        prefix = (
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8.CompressedTensorsW8A8Int8."
        )
        self.assertEqual(
            {target for target in hooks if target.startswith(prefix)},
            {prefix + name for name in (
                "__init__", "process_weights_after_loading", "apply_weights"
            )},
        )
        callback, hook_type = hooks[prefix + "apply_weights"]
        self.assertIs(hook_type, HookType.AROUND)
        original = Mock(return_value=sentinel.original)
        with (
            patch.object(compressed_int8, "is_hcu", return_value=True),
            patch.object(compressed_int8, "apply_weights", return_value=sentinel.hcu) as hcu,
        ):
            result = callback(original, sentinel.scheme, sentinel.layer, sentinel.x, None)
        self.assertIs(result, sentinel.hcu)
        hcu.assert_called_once_with(
            sentinel.scheme, sentinel.layer, sentinel.x, None, None, None
        )
        original.assert_not_called()

    @unittest.skipUnless(_DRAFT_CONFIG.is_file(), "W8A8 DSpark checkpoint is unavailable")
    def test_checkpoint_selects_compressed_tensors_int8_linear(self):
        from hcu_sglang.kimi31_dspark import normalize_speculator
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
            CompressedTensorsW8A8Int8,
        )

        config = normalize_speculator(json.loads(_DRAFT_CONFIG.read_text()))
        self.assertEqual(config.architectures, ["DSparkDraftModel"])
        self.assertEqual(config.compression_config["quant_method"], "compressed-tensors")
        quant = CompressedTensorsConfig.from_config(config.compression_config)
        layer = Mock(spec=LinearBase)
        with patch.object(quant, "_check_scheme_supported"):
            method = quant.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertIsInstance(layer.scheme, CompressedTensorsW8A8Int8)
        self.assertIs(method.quantization_config, quant)


if __name__ == "__main__":
    unittest.main()

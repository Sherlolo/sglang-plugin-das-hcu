from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as Q

class AiterActivationTests(unittest.TestCase):
    def test_public_apply_preserves_activation_and_situ_parameters(self):
        for activation in ("situ", "silu"):
            with self.subTest(activation=activation):
                method=object.__new__(Q.SlimQuantW4A8Int8AiterMoEMethod)
                method.moe_runner_config=S(activation=activation,gemm1_alpha=4.,gemm1_clamp_limit=25.,num_experts=2,routed_scaling_factor=1.)
                layer=S(w13_weight=torch.zeros(2,16,4,dtype=torch.int8),w2_weight=torch.zeros(2,8,4,dtype=torch.int8),w13_weight_scale=torch.ones(2,16,1),w2_weight_scale=torch.ones(2,8,1))
                x=torch.randn(3,8);ids=torch.tensor([[0,1]]*3);weights=torch.full((3,2),.5)
                config=S(solution_type="test")
                with patch.object(Q,"_lookup_aiter_w4a8_config",return_value=(True,config)),patch.object(Q,"get_parallel",return_value=S(moe_ep_size=1)),patch("sglang.srt.layers.moe.utils.get_moe_runner_backend",return_value=S(is_aiter=lambda:True)),patch("aiter.moe.aiter_moe",return_value=x) as kernel:
                    out=method.apply(layer,S(hidden_states=x,topk_output=(weights,ids,None)))
                self.assertIs(out.hidden_states,x)
                kwargs=kernel.call_args.kwargs
                self.assertEqual(kwargs["activation"],activation)
                self.assertEqual(kwargs["gemm1_alpha"],4. if activation=="situ" else None)
                self.assertEqual(kwargs["gemm1_limit"],25. if activation=="situ" else None)
                self.assertEqual(kwargs["topk_ids"].dtype,torch.int32)
                self.assertIs(kwargs["moe_config"],config)

if __name__=="__main__":unittest.main()

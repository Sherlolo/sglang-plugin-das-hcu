import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
from sglang.srt.layers.moe.utils import DeepEPMode


class Kimi31NcclEPDispatcher(BaseDispatcher):
    _logged = False

    def __init__(self, group, router_topk, num_experts, num_local_experts,
                 hidden_size, params_dtype=None, deepep_mode=DeepEPMode.NORMAL,
                 permute_fusion=False, async_finish=False, return_recv_hook=False,
                 runner_backend=None):
        super().__init__()
        from sglang.srt.batch_overlap.two_batch_overlap import is_tbo_enabled

        if not deepep_mode.is_normal() or is_tbo_enabled():
            raise ValueError("Kimi31 NCCL EP supports eager normal dispatch only")
        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.runner_backend = runner_backend
        self.expert_mask_gpu = None
        self._pending = None
        if num_experts != self.world * num_local_experts:
            raise ValueError("Kimi31 NCCL EP requires equally partitioned experts")
        if not type(self)._logged:
            logging.getLogger(__name__).info(
                "Kimi31 EP transport=NCCL all-gather/reduce-scatter; EP=%s, local experts=%s; compute=DeepGEMM W4A8 HIPC",
                self.world, num_local_experts,
            )
            type(self)._logged = True

    def _gather(self, x, capacity, pad=0):
        padded = F.pad(x, (0, 0, 0, capacity - x.shape[0]), value=pad).contiguous()
        gathered = torch.empty((self.world * capacity, *x.shape[1:]),
                               dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, padded, group=self.group)
        return gathered

    def dispatch(self, hidden_states, topk_output):
        from lightop.quant import per_token_quant_int8

        if self._pending is not None:
            raise RuntimeError("Kimi31 NCCL EP dispatch has an unfinished combine")
        if not self.quant_config.get('hcu_w4a8_int8_dispatch'):
            raise ValueError("Kimi31 NCCL EP requires the W4A8 INT8 quantization contract")
        count = hidden_states.shape[0]
        size = torch.tensor([count], dtype=torch.int64, device=hidden_states.device)
        sizes = torch.empty(self.world, dtype=torch.int64, device=hidden_states.device)
        dist.all_gather_into_tensor(sizes, size, group=self.group)
        capacity = int(sizes.max().item())
        self._pending = (count, capacity)
        if capacity == 0:
            return DeepEPNormalDispatchOutput(
                hidden_states.to(torch.int8),
                torch.empty((0, 1), dtype=torch.float32, device=hidden_states.device),
                topk_output.topk_ids.to(torch.int64), topk_output.topk_weights,
                [0] * self.num_local_experts,
            )
        if count:
            qx, scale = per_token_quant_int8(hidden_states)
        else:
            qx = hidden_states.to(torch.int8)
            scale = torch.empty((0, 1), dtype=torch.float32, device=hidden_states.device)
        x = self._gather(qx, capacity)
        scale = self._gather(scale, capacity)
        ids = self._gather(topk_output.topk_ids.to(torch.int64), capacity, -1)
        weights = self._gather(topk_output.topk_weights.float(), capacity)
        start = self.rank * self.num_local_experts
        local = (ids >= start) & (ids < start + self.num_local_experts)
        ids = torch.where(local, ids - start, -1)
        counts = torch.bincount(ids[local], minlength=self.num_local_experts)
        counts = (((counts + 255) // 256) * 256).cpu().tolist()
        return DeepEPNormalDispatchOutput(x, scale, ids, weights, counts)

    def combine(self, combine_input):
        if self._pending is None:
            raise RuntimeError("Kimi31 NCCL EP combine has no dispatch")
        count, capacity = self._pending
        x = combine_input.hidden_states.contiguous()
        if capacity == 0:
            output = x
        else:
            if x.shape != (self.world * capacity, self.hidden_size):
                raise ValueError("Kimi31 NCCL EP combine row layout mismatch")
            output = torch.empty((capacity, self.hidden_size), dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(output, x, group=self.group)
        self._pending = None
        return output[:count]

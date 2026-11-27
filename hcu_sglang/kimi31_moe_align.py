import logging
import os


_logged_routes = set()


def should_use_lightop_align(topk_ids, num_experts, block_size):
    if os.environ.get("HCU_KIMI31_LIGHTOP_MOE_ALIGN") != "1":
        return False
    max_lanes = int(os.environ.get("HCU_KIMI31_LIGHTOP_MOE_ALIGN_MAX_LANES", "1024"))
    if max_lanes < 1:
        raise ValueError("HCU_KIMI31_LIGHTOP_MOE_ALIGN_MAX_LANES must be positive")
    selected = (num_experts in (449, 897) and block_size in (16, 64)
                and 0 < topk_ids.numel() <= min(max_lanes, 1024))
    route = (topk_ids.numel(), num_experts, block_size, selected)
    if route not in _logged_routes:
        logging.getLogger(__name__).info(
            "Kimi31 MoE align=%s lanes=%d experts=%d block=%d verified_max_lanes=1024",
            "LightOp" if selected else "baseline", *route[:3],
        )
        _logged_routes.add(route)
    return selected


def align_hcu_tokens(original, topk_ids, num_experts, block_size,
                     sorted_ids, expert_ids, num_tokens_post_pad):
    args = (topk_ids, num_experts, block_size, sorted_ids, expert_ids,
            num_tokens_post_pad)
    if not should_use_lightop_align(topk_ids, num_experts, block_size):
        return original(*args)
    from lightop import moe

    align_op = getattr(moe, "moe_align_block_size_out", None)
    if align_op is None:
        align_op = moe.moe_align_block_size
    return align_op(*args, None, None, None, False, True)


def around_boltop_align(original, topk_ids, block_size, num_experts, expert_map=None):
    if not should_use_lightop_align(topk_ids, num_experts, block_size):
        return original(topk_ids, block_size, num_experts, expert_map)
    import torch

    count = topk_ids.numel()
    capacity = ((count + num_experts * (block_size - 1) + 3) // 4) * 4
    sorted_ids = torch.full((capacity,), count, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros(((capacity + block_size - 1) // block_size,),
                             dtype=torch.int32, device=topk_ids.device)
    padded = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    align_hcu_tokens(None, topk_ids, num_experts, block_size, sorted_ids, expert_ids, padded)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]
    return sorted_ids, expert_ids, padded


def register_hooks():
    if os.environ.get("HCU_KIMI31_LIGHTOP_MOE_ALIGN") != "1":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "boltops.fused_moe.triton.fused_moe.moe_align_block_size",
        around_boltop_align, HookType.AROUND,
    )

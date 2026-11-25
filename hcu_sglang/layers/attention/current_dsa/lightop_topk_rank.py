# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _rank_selected(scores, selected, result, WIDTH: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, K)
    ids = tl.load(selected + row * K + pos)
    valid_id = (ids >= 0) & (ids < WIDTH)
    values = tl.load(scores + row * WIDTH + ids, valid_id, other=-float("inf"))
    bits = values.to(tl.uint32, bitcast=True)
    # Linearize IEEE-754 float order into unsigned order. The low 32 bits
    # retain the input position for stable ties and the 18-bit token ID.
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    payload = ((K - 1 - pos).to(tl.uint32) << 18) | (ids + 1).to(tl.uint32)
    key = (ordered.to(tl.uint64) << 32) | payload.to(tl.uint64)
    key = tl.where(valid_id & (values > -float("inf")), key, 0)
    sorted_keys = tl.sort(key, descending=True)
    chosen = (sorted_keys & 0x3FFFF).to(tl.int32) - 1
    tl.store(result + row * K + pos, chosen)


def rank_selected(score, chosen):
    rows, width = score.shape
    k = chosen.shape[1]
    result = torch.empty_like(chosen)
    _rank_selected[(rows,)](score, chosen, result, width, k, num_warps=4)
    return result

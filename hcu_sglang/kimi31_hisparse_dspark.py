"""Connect static Kimi DSpark verify to the HiSparse slot lifecycle."""

import hashlib
import os
import tempfile
from functools import cache
from pathlib import Path


_SPEC_HEADER = "kvcacheio/hisparse_spec.cuh"


@cache
def hcu_spec_header():
    from sglang.kernels.ops.kvcache import hisparse

    kernels = Path(hisparse.__file__).resolve().parents[2]
    source = (kernels / "jit/csrc" / _SPEC_HEADER).read_text()
    old = "LaunchKernel(dim3(static_cast<uint32_t>(bs)), 512, device)"
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the HiSparse speculative commit launch.")
    # The gfx compiler limits this variant to 256 threads. Its loops use blockDim.x.
    patched = source.replace(old, old.replace("512", "256"))
    digest = hashlib.sha256(patched.encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"hcu-hisparse-{os.getuid()}" / digest
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    header = directory / Path(_SPEC_HEADER).name
    if not header.is_file():
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as out:
            out.write(patched)
        os.replace(out.name, header)
    return str(header)


def load_hcu_spec_jit(original, name, *args, **kwargs):
    if name == "hisparse_spec" and kwargs.get("cuda_files") == [_SPEC_HEADER]:
        import torch

        if torch.version.hip is not None:
            kwargs["cuda_files"] = [hcu_spec_header()]
    return original(name, *args, **kwargs)


def bind_spec_host_cache(_result, self, coordinator, *, num_draft_tokens):
    import torch

    pool = coordinator.mem_pool_host
    pointers = getattr(pool, "data_ptrs", None)
    if not self.enabled or torch.version.hip is None or pointers is None:
        return
    # ROCm kernels must use the GPU mapping of the registered CPU buffer.
    # The host pool already resolves this address for its backup kernels.
    bindings = torch.empty((pointers.numel(), 2), dtype=torch.int64, device=pointers.device)
    bindings[:, 0].copy_(pointers.view(torch.int64))
    bindings[:, 1].fill_(pool.token_stride_size)
    self._hcu_host_bindings = bindings


def swap_in_with_host_binding(original, self, req_pool_indices, compressed_seq_lens,
                              top_k_result, layer_id, **plan):
    bindings = getattr(self, "_hcu_host_bindings", None)
    if bindings is not None:
        plan.setdefault("host_binding", bindings[layer_id])
    return original(
        self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, **plan
    )


def validate_static_hisparse(server_args):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if not (cfg.enable_hisparse and (cfg.speculative_algorithm or "").upper() == "DSPARK"):
        return
    if os.environ.get("SGLANG_RAGGED_VERIFY_MODE") != "static":
        raise ValueError("DSpark HiSparse requires SGLANG_RAGGED_VERIFY_MODE=static.")
    if not cfg.disable_overlap_schedule:
        raise ValueError("DSpark HiSparse requires --disable-overlap-schedule.")
    if cfg.speculative_dspark_align_verify_tokens_to_graph_tier:
        raise ValueError("DSpark HiSparse requires a fixed verify width per request.")


def make_staged_spec_input(original, self, future_indices):
    if not self.spec_algo.is_dspark():
        return original(self, future_indices)

    # The synchronous scheduler seeds the concrete state in its staging hook.
    return None


def seed_staged_decode(batch, scheduler, reqs):
    if not batch.spec_algorithm.is_dspark():
        return

    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

    bonus = scheduler.future_map.output_tokens_buf[batch.req_pool_indices]
    draft = make_draft_input_v2(bonus_tokens=bonus, new_seq_lens=batch.seq_lens)
    draft.nxt_kv_lens_cpu = batch.seq_lens_cpu
    draft.nxt_kv_lens_sum = batch.seq_lens_sum
    batch.spec_info = draft


def prepare_verify(self, *, batch, verify_input, **kwargs):
    coordinator = batch.hisparse_coordinator
    if coordinator is None:
        return
    if verify_input.ragged_verify_layout is not None:
        raise RuntimeError("DSpark HiSparse requires a static verify layout.")
    coordinator.prepare_spec_verify(batch)


def commit_verify(self, *, batch, seq_lens_pre_verify, seq_lens_post_verify, commit_lens):
    import torch

    coordinator = batch.hisparse_coordinator
    if coordinator is None:
        return
    width = self.verify_num_draft_tokens
    steps = torch.arange(width, device=commit_lens.device)
    rows = torch.arange(commit_lens.numel(), device=commit_lens.device)[:, None]
    indices = torch.where(steps < commit_lens[:, None], rows * width + steps, -1)
    coordinator.commit_spec_accept_tokens(batch=batch, accept_indices=indices)


def register_hooks():
    from sglang.srt.arg_groups import hisparse_hook
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.hisparse_hook.validate_hisparse",
        validate_static_hisparse, HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.managers.overlap_utils.FutureMap.make_staged_spec_input",
        make_staged_spec_input, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler._build_hisparse_decode_batch",
        seed_staged_decode, HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.kernels.ops.kvcache.hisparse.load_jit",
        load_hcu_spec_jit, HookType.AROUND,
    )
    swap_manager = "sglang.srt.managers.hisparse_coordinator.HiSparseSpecSwapManager"
    HookRegistry.register(
        f"{swap_manager}.__init__", bind_spec_host_cache, HookType.AFTER,
    )
    HookRegistry.register(
        f"{swap_manager}.swap_in", swap_in_with_host_binding, HookType.AROUND,
    )
    prefix = "sglang.srt.speculative.dspark_components"
    HookRegistry.register(
        f"{prefix}.dspark_verify.TargetVerifyExecutor._forward_prepared_verify",
        prepare_verify, HookType.BEFORE,
    )
    HookRegistry.register(
        f"{prefix}.dspark_worker_v2.DSparkWorkerV2._commit_target_mamba_states_after_verify",
        commit_verify, HookType.BEFORE,
    )
    if "DSPARK" not in hisparse_hook.HISPARSE_VALIDATED_SPEC_ALGORITHMS:
        hisparse_hook.HISPARSE_VALIDATED_SPEC_ALGORITHMS += ("DSPARK",)

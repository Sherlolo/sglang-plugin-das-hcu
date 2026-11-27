import os


def _enabled(name):
    return os.environ.get(name, '0').lower() in ('1', 'true')


def around_dispatch(original, group, device, **kwargs):
    from hcu_sglang.config import resolved_custom_all_reduce_backend
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(group, device, **kwargs)
    backend = resolved_custom_all_reduce_backend()
    if backend == 'auto':
        return original(group, device, **kwargs)
    if backend == 'aiter':
        from hcu_adapter import hcu_env, kill_switch
        policy, _ = hcu_env.resolve('HCU_AITER_AR')
        if kill_switch.op_disabled('aiter_ar') or str(policy).startswith('refused'):
            raise RuntimeError('HCU adapter policy disabled AITER all-reduce; use RCCL fallback')
    from hcu_sglang.distributed_aiter import dispatch_custom_allreduce
    return dispatch_custom_allreduce(group, device, backend=backend)


def _communicator(self, input, method):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not _enabled('SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM'):
        return None
    communicator = getattr(self, 'ca_comm', None)
    if (communicator is None or communicator.disabled
            or not callable(getattr(communicator, method, None))):
        return None
    return communicator


def _shares_storage(input, output):
    if input is output:
        return True
    if not hasattr(input, 'untyped_storage') or not hasattr(output, 'untyped_storage'):
        return False
    return input.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()


def around_reduce_scatter(original, self, output, input):
    communicator = _communicator(self, input, 'reduce_scatter')
    if communicator is None or _shares_storage(input, output) or not communicator.should_custom_ar(input):
        return original(self, output, input)
    communicator.reduce_scatter(input, output, registered=False)
    return output


def around_all_gather(original, self, output, input):
    communicator = _communicator(self, input, 'all_gather_unreg')
    if (communicator is None or _shares_storage(input, output) or input.numel() > 256 * 6144
            or not communicator.should_custom_ag(input)):
        return original(self, output, input)
    if hasattr(input, 'view') and hasattr(output, 'view'):
        if (not input.is_contiguous() or not output.is_contiguous()
                or input.dtype != output.dtype or input.device != output.device
                or output.numel() != input.numel() * self.world_size):
            return original(self, output, input)
        communicator.all_gather_unreg(input.view(-1), out=output.view(-1))
    else:
        communicator.all_gather_unreg(input, out=output)
    return None


def register_hooks():
    if not (_enabled('SGLANG_USE_AITER_AR')
            or _enabled('SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM')):
        return
    from hcu_sglang.environ import register_envs
    register_envs()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        'sglang.srt.distributed.device_communicators.custom_all_reduce.dispatch_custom_allreduce',
        around_dispatch, HookType.AROUND,
    )
    base = 'sglang.srt.distributed.parallel_state.GroupCoordinator.'
    HookRegistry.register(base + '_reduce_scatter_tensor', around_reduce_scatter, HookType.AROUND)
    HookRegistry.register(base + '_all_gather_into_tensor', around_all_gather, HookType.AROUND)

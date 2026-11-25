"""Keep DSpark's independent KV cache addressable by HiSparse logical IDs."""
import logging

logger = logging.getLogger(__name__)


def _ratio():
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config
    return parse_hisparse_config().host_to_device_ratio


def charge_draft_logical_rows(_result, configurator):
    if not configurator.is_draft_worker and configurator.spec_algorithm.is_dspark():
        configurator.dflash_draft_loc_space_scale *= _ratio()


def allocate_draft_logical_rows(sizes, configurator, *, config):
    if not configurator.is_draft_worker or not configurator.spec_algorithm.is_dspark():
        return sizes
    import msgspec
    rows = sizes.max_total_num_tokens * _ratio()
    logger.info("HiSparse DSpark draft KV uses %d logical rows with target page size %d",
                rows, configurator.pool_page_size)
    return msgspec.structs.replace(sizes, max_total_num_tokens=rows)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    logger.info("HiSparse + DSpark logical KV hooks enabled from %s", __file__)
    target = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    HookRegistry.register(f"{target}.__post_init__", charge_draft_logical_rows, HookType.AFTER)
    HookRegistry.register(f"{target}._derive_pool_sizes", allocate_draft_logical_rows, HookType.AFTER)

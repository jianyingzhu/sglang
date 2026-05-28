"""Registry for pluggable RadixCache factories.

If `--radix-cache-backend` is unset (by default), the built-in selection
chain is used to pick a cache implementation.

To plug in a custom backend, register it under a string name via
`register_radix_cache_backend(name, factory)`, then select it with
`--radix-cache-backend <name>` (the flag accepts only registered names).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class TreeCacheBuildContext:
    """Radix Cache construction arguments."""

    server_args: ServerArgs
    params: CacheInitParams
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    enable_hierarchical_cache: bool
    disable_radix_cache: bool
    effective_chunked_prefill_size: Optional[int]
    tp_worker: Any
    model_config: ModelConfig
    tp_size: int
    tp_rank: int
    tp_group: Any
    attn_tp_group: Any = None
    attn_cp_group: Any = None
    pp_group: Any = None
    dp_rank: Optional[int] = None
    attn_cp_rank: int = 0


RadixCacheFactory = Callable[[TreeCacheBuildContext], BasePrefixCache]

_RADIX_CACHE_REGISTRY: dict[str, RadixCacheFactory] = {}


def register_radix_cache_backend(name: str, factory: RadixCacheFactory) -> None:
    """Register a radix-cache factory under `name`.

    Raises ValueError if `name` is empty/whitespace-only or already
    registered.
    """
    if not name.strip():
        raise ValueError(
            f"register_radix_cache_backend: name must be non-empty, got {name!r}"
        )
    if name in _RADIX_CACHE_REGISTRY:
        raise ValueError(
            f"register_radix_cache_backend: {name!r} is already registered"
        )
    _RADIX_CACHE_REGISTRY[name] = factory


def get_radix_cache_factory(name: str) -> Optional[RadixCacheFactory]:
    return _RADIX_CACHE_REGISTRY.get(name)


def registered_radix_cache_backends() -> list[str]:
    return list(_RADIX_CACHE_REGISTRY.keys())


def default_radix_cache_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Built-in Radix Cache selection chain."""
    server_args = ctx.server_args
    params = ctx.params

    if ctx.effective_chunked_prefill_size is not None and ctx.disable_radix_cache:
        if not ctx.is_hybrid_swa:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache

            return ChunkCache(params)
        from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

        return SWAChunkCache(params)

    if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
        # lazy import to avoid JIT overhead
        from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

        logger.info("Using experimental C++ radix tree implementation.")
        return RadixCacheCpp(params=params, server_args=server_args)

    if envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get():
        from sglang.srt.mem_cache.unified_cache_components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        tree_components = [ComponentType.FULL]
        if ctx.is_hybrid_swa or ctx.is_hybrid_ssm:
            tree_components.append(
                ComponentType.SWA if ctx.is_hybrid_swa else ComponentType.MAMBA
            )
        params.tree_components = tuple(tree_components)
        cache = UnifiedRadixCache(params)
        if ctx.enable_hierarchical_cache:
            cache.init_hicache(server_args, params)
            ctx.tp_worker.register_hicache_layer_transfer_counter(
                cache.cache_controller.layer_done_counter
            )
        return cache

    if ctx.enable_hierarchical_cache:
        if ctx.is_hybrid_ssm:
            from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

            cache = HiMambaRadixCache(params=params, server_args=server_args)
        else:
            from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

            cache = HiRadixCache(params=params, server_args=server_args)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
        return cache

    if ctx.is_hybrid_swa:
        from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

        return SWARadixCache(params=params)

    if ctx.is_hybrid_ssm:
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        return MambaRadixCache(params)

    if server_args.enable_lmcache:
        from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
            LMCRadixCache,
        )

        return LMCRadixCache(
            params=params,
            model_config=ctx.model_config,
            tp_size=ctx.tp_size,
            rank=ctx.tp_rank,
            tp_group=ctx.tp_group,
        )

    from sglang.srt.mem_cache.radix_cache import RadixCache

    return RadixCache(params)


def create_tree_cache(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Route to the matching factory to construct Radix Cache."""
    name = ctx.server_args.radix_cache_backend
    if name:
        factory = get_radix_cache_factory(name)
        if factory is None:
            raise ValueError(
                f"--radix-cache-backend={name!r} is not registered. "
                f"Registered backends: {registered_radix_cache_backends()}. "
                "External backends must call register_radix_cache_backend(...) at import time."
            )
        cache = factory(ctx)
        source = f"registered({name!r})"
    else:
        cache = default_radix_cache_factory(ctx)
        source = "default"

    # -- KV Connector wrapping --
    # If --kv-connector-cls is set, wrap the cache with ExtendedRadixCache
    # which adds external KV storage capabilities (e.g., FlexKV).
    # This works with ANY inner cache (RadixCache, SWARadixCache, etc.)
    kv_connector_cls_name = ctx.server_args.kv_connector_cls
    if kv_connector_cls_name is not None:
        connector = _load_and_create_connector(kv_connector_cls_name, ctx)
        if connector is not None:
            from sglang.srt.mem_cache.extended_radix_cache import ExtendedRadixCache

            cache = ExtendedRadixCache(
                params=ctx.params,
                connector=connector,
                inner_cache=cache,
            )
            source = f"{source}+ExtendedRadixCache({kv_connector_cls_name})"
            logger.info(
                "KV connector enabled: cls=%s, wrapping cache with ExtendedRadixCache",
                kv_connector_cls_name,
            )

    streaming_wrapped = False
    if (
        ctx.server_args.enable_streaming_session
        and not cache.supports_streaming_session()
    ):
        from sglang.srt.session.streaming_session import StreamingSession

        cache = StreamingSession(cache)
        streaming_wrapped = True

    logger.info(
        "Tree cache initialized: source=%s impl=%s hybrid_swa=%s hybrid_ssm=%s "
        "hierarchical=%s streaming_wrapped=%s",
        source,
        type(cache).__name__,
        ctx.is_hybrid_swa,
        ctx.is_hybrid_ssm,
        ctx.enable_hierarchical_cache,
        streaming_wrapped,
    )
    return cache


def _load_and_create_connector(
    cls_name: str, ctx: TreeCacheBuildContext
) -> Optional["BaseKVConnector"]:
    """Dynamically load and instantiate the KV connector class.

    Args:
        cls_name: Fully qualified class name or a short alias.
            Short aliases:
              "flexkv" → sglang.srt.mem_cache.storage.flexkv.flexkv_connector.FlexKVConnector
            Full paths: e.g. "my_module.MyConnector"
        ctx: Build context with server_args, params, etc.

    Returns:
        Instantiated connector, or None if loading fails.
    """
    import importlib

    # Short-name registry for convenience (--kv-connector-cls flexkv)
    _CONNECTOR_ALIASES = {
        "flexkv": "sglang.srt.mem_cache.storage.flexkv.flexkv_connector.FlexKVConnector",
        "FlexKV": "sglang.srt.mem_cache.storage.flexkv.flexkv_connector.FlexKVConnector",
        "lmcache": "sglang.srt.mem_cache.storage.lmcache.lmc_connector.LMCacheConnector",
    }

    # Resolve alias if applicable
    resolved_name = _CONNECTOR_ALIASES.get(cls_name, cls_name)

    try:
        module_path, class_name = resolved_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        connector_cls = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(
            "Failed to load kv_connector_cls=%r (resolved=%r): %s. "
            "Falling back to no connector.",
            cls_name, resolved_name, e,
        )
        return None

    try:
        connector = connector_cls(
            params=ctx.params,
            server_args=ctx.server_args,
            tp_rank=ctx.tp_rank,
            dp_rank=ctx.dp_rank,
            attn_cp_rank=ctx.attn_cp_rank,
            pp_group=ctx.pp_group,
            attn_tp_group=ctx.attn_tp_group,
            attn_cp_group=ctx.attn_cp_group,
        )
    except Exception as e:
        logger.error(
            "Failed to instantiate kv_connector %s: %s. "
            "Falling back to no connector.",
            cls_name, e,
        )
        return None

    return connector

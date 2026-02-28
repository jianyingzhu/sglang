from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class ExtRadixCache(BasePrefixCache):
    """RadixCache extended with an optional external KV storage connector.

    When *connector* is ``None`` every method delegates directly to the
    internal ``RadixCache`` instance, giving behaviour identical to a plain
    ``RadixCache``.  When a ``BaseKVConnector`` is supplied, additional
    storage-level match / load / store logic is layered on top.
    """

    def __init__(
        self,
        params: CacheInitParams,
        connector: Optional[BaseKVConnector] = None,
    ):
        self._radix = RadixCache(params)
        self._connector = connector

        self._inflight_reqid2node: Dict[int, TreeNode] = {}
        self._pending_load_info: Dict[str, tuple] = {}
        self._load_queue: List[LoadOperation] = []
        self._ongoing_load_back: Dict[int, Tuple[TreeNode, int]] = {}

    # ------------------------------------------------------------------
    # PrefixCacheTrait attributes – delegate to inner RadixCache
    # ------------------------------------------------------------------

    @property
    def req_to_token_pool(self):
        return self._radix.req_to_token_pool

    @property
    def token_to_kv_pool_allocator(self):
        return self._radix.token_to_kv_pool_allocator

    @property
    def page_size(self):
        return self._radix.page_size

    @property
    def disable(self):
        return self._radix.disable

    @property
    def device(self):
        return self._radix.device

    @property
    def metrics_collector(self):
        return self._radix.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self._radix.metrics_collector = value

    # ------------------------------------------------------------------
    # Layer transfer property – exposed for scheduler
    # ------------------------------------------------------------------

    @property
    def layer_done_counter(self):
        if self._connector is None:
            return None
        return self._connector.layer_done_counter

    # ------------------------------------------------------------------
    # Pure delegation (behaviour identical to RadixCache)
    # ------------------------------------------------------------------

    def reset(self):
        if self._connector is not None:
            for _, node in self._inflight_reqid2node.items():
                self._radix.dec_lock_ref(node)
            self._inflight_reqid2node.clear()

            for _, (node, _) in self._ongoing_load_back.items():
                self._radix.dec_lock_ref(node)
            self._ongoing_load_back.clear()

            self._load_queue.clear()
            self._pending_load_info.clear()
            self._connector.reset()

        self._radix.reset()

    def insert(self, key, value=None, **kwargs):
        return self._radix.insert(key, value=value, **kwargs)

    def inc_lock_ref(self, node):
        return self._radix.inc_lock_ref(node)

    def dec_lock_ref(self, node, swa_uuid_for_lock=None):
        return self._radix.dec_lock_ref(node, swa_uuid_for_lock)

    def evictable_size(self):
        return self._radix.evictable_size()

    def protected_size(self):
        return self._radix.protected_size()

    def total_size(self):
        return self._radix.total_size()

    def pretty_print(self):
        return self._radix.pretty_print()

    def all_values_flatten(self):
        return self._radix.all_values_flatten()

    def take_events(self):
        return self._radix.take_events()

    def cache_unfinished_req(self, req: Req, **kwargs):
        return self._radix.cache_unfinished_req(req, **kwargs)

    # ------------------------------------------------------------------
    # Extended methods (connector-aware)
    # ------------------------------------------------------------------

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        base_res = self._radix.match_prefix(key, **kwargs)
        if self._connector is None:
            return base_res

        value: torch.Tensor = base_res.device_indices
        last_node = base_res.last_device_node

        uncached_len = len(key) - value.numel()
        if uncached_len <= 0:
            return base_res

        token_mask = torch.zeros(len(key), dtype=torch.bool)
        token_mask[value.numel():] = True

        storage_res = self._connector.match_storage(
            token_ids=key.token_ids,
            token_mask=token_mask,
        )

        rid = kwargs.get("rid")
        if storage_res.hit_length > 0 and rid is not None:
            self._pending_load_info[rid] = (
                storage_res.task_id,
                key,
                value.numel(),
            )

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            host_hit_length=storage_res.hit_length,
        )

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)

        if self._connector is None or host_hit_length <= 0:
            return empty, last_node

        rid = kwargs.get("rid")
        if rid is None or rid not in self._pending_load_info:
            return empty, last_node

        task_id, key, gpu_cached_len = self._pending_load_info.pop(rid)

        if mem_quota is not None and host_hit_length > mem_quota:
            return empty, last_node

        device_indices = self._radix.token_to_kv_pool_allocator.alloc(
            host_hit_length
        )
        if device_indices is None:
            self.evict(host_hit_length)
            device_indices = self._radix.token_to_kv_pool_allocator.alloc(
                host_hit_length
            )
        if device_indices is None:
            logger.warning(
                "Failed to allocate %d GPU slots for external load",
                host_hit_length,
            )
            return empty, last_node

        start = gpu_cached_len
        end = start + host_hit_length
        new_node = self._radix.insert_node_with_value(
            parent=last_node,
            key=key[start:end],
            value=device_indices,
        )

        self._radix.inc_lock_ref(new_node)

        self._load_queue.append(
            LoadOperation(
                task_id=task_id,
                device_indices=device_indices,
                tag=(new_node.id, new_node),
            )
        )

        return device_indices, new_node

    def ready_to_load_host_cache(self) -> int:
        if self._connector is None:
            return -1

        if not self._connector.worker_connected:
            raise RuntimeError(
                "External KV connector worker not connected"
            )

        counter = self._connector.layer_done_counter
        if counter is None:
            raise RuntimeError("Layer done counter not available")

        if not self._load_queue:
            return -1

        producer_id = counter.update_producer()
        counter.events[producer_id].reset_for_new_transfer()

        for op in self._load_queue:
            node_id, node = op.tag
            self._ongoing_load_back[node_id] = (node, producer_id)

        self._connector.launch_load_batch(self._load_queue, producer_id)
        self._load_queue.clear()

        return producer_id

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        self._radix.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if self._connector is None:
            return

        if req.req_pool_idx is None and not is_insert:
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self._radix.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_last_node = req.last_node
        if new_last_node is None:
            return
        
        self._connector.store_kv_async(
            token_ids=token_ids,
            kv_indices=kv_indices,
            req_id=req.req_pool_idx,
        )

        if req.req_pool_idx in self._inflight_reqid2node:
            self._radix.dec_lock_ref(
                self._inflight_reqid2node[req.req_pool_idx]
            )

        self._radix.inc_lock_ref(new_last_node)
        self._inflight_reqid2node[req.req_pool_idx] = new_last_node

    def evict(self, num_tokens: int):
        if self._radix.disable:
            return

        if self._connector is not None:
            if num_tokens > self._radix.evictable_size():
                completed_ids = self._connector.sync_writes_for_eviction(num_tokens, self._radix)
                for req_id in completed_ids:
                    node = self._inflight_reqid2node.pop(req_id, None)
                    if node is not None:
                        self._radix.dec_lock_ref(node)

        self._radix.evict(num_tokens)

    def check_kv_events(self):
        if self._connector is None:
            return
        self._writing_check()
        self._loading_check()

    def prefetch(self, req: Req) -> None:
        return

    def can_be_scheduled(self, req: Req) -> bool:
        return True

    def release_aborted_request(self, req: Req) -> None:
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _writing_check(self) -> None:
        completed = self._connector.poll_completed_stores()
        skipped = self._connector.poll_skipped_stores()
        for req_id in completed + skipped:
            node = self._inflight_reqid2node.pop(req_id, None)
            if node is not None:
                self._radix.dec_lock_ref(node)

    def _loading_check(self) -> None:
        counter = self._connector.layer_done_counter
        if counter is None or not self._ongoing_load_back:
            return

        completed_ids = []
        for node_id, (node, producer_id) in self._ongoing_load_back.items():
            event = counter.events[producer_id]
            if event._finished:
                completed_ids.append(node_id)

        for node_id in completed_ids:
            node, _ = self._ongoing_load_back.pop(node_id)
            self._radix.dec_lock_ref(node)

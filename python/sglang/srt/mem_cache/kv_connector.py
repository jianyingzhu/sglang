from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional

import torch


class MatchStorageResult(NamedTuple):
    """Result of querying external KV storage for cached tokens."""

    hit_length: int
    task_id: int  # -1 if no match or not applicable


class LoadOperation(NamedTuple):
    """A pending load operation from external storage to GPU."""

    task_id: int
    device_indices: torch.Tensor
    tag: Any  # opaque tag for the caller to track (e.g. node_id)


class BaseKVConnector(ABC):
    """Abstract interface for external KV cache storage connectors.

    Implementations must not depend on TreeNode, RadixCache, or any radix tree
    internals.  All interactions happen through token_ids, kv_indices, and
    task_id primitives.
    """

    # ------------------------------------------------------------------
    # Query / Load
    # ------------------------------------------------------------------

    @abstractmethod
    def match_storage(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
    ) -> MatchStorageResult:
        """Check how many tokens are available in external storage.

        Args:
            token_ids: Full token id sequence.
            token_mask: Boolean mask – True for positions to check.

        Returns:
            MatchStorageResult with hit_length and task_id for later load.
        """
        ...

    @abstractmethod
    def launch_load_batch(
        self,
        load_ops: List[LoadOperation],
        producer_id: int,
    ) -> int:
        """Launch a batch of load operations (layer-by-layer transfer).

        Returns:
            Number of operations actually launched.
        """
        ...

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @abstractmethod
    def store_kv_async(
        self,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        req_id: int,
    ) -> int:
        """Asynchronously store KV cache for *token_ids* at *kv_indices*.

        Returns:
            task_id (>= 0) if a real task was launched, -1 otherwise.
        """
        ...

    @abstractmethod
    def poll_completed_stores(self) -> List[int]:
        """Non-blocking poll returning req_ids whose stores completed."""
        ...

    @abstractmethod
    def poll_skipped_stores(self) -> List[int]:
        """Non-blocking poll returning req_ids whose stores were skipped."""
        ...

    @abstractmethod
    def wait_all_inflight(self) -> None:
        """Block until every in-flight store finishes."""
        ...

    @abstractmethod
    def sync_writes_for_eviction(self, num_tokens: int, radix_cache: Any) -> List[int]:
        """Synchronize in-flight store operations for eviction.

        Called when GPU eviction alone cannot free enough tokens.  The connector
        should try to complete enough writes to unlock at least *num_tokens*
        worth of cache entries.  The concrete strategy (poll, block, partial
        wait, etc.) is entirely up to the implementation.

        Args:
            num_tokens: Number of tokens that still need to be freed.
            radix_cache: The RadixCache instance, can be used to query
                evictable_size() during synchronization.

        Returns:
            List of req_ids whose stores have completed.
        """
        ...

    # ------------------------------------------------------------------
    # Layer transfer
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def layer_done_counter(self) -> Any:
        """Return the layer-wise transfer counter, or None."""
        ...

    @property
    @abstractmethod
    def worker_connected(self) -> bool:
        """Whether the layer-wise transfer worker is connected."""
        ...

    @abstractmethod
    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        """Register the layer transfer counter with the KV cache pool."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset connector state (called on cache reset)."""
        pass

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

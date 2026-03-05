from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional

import torch


class LoadOperation(NamedTuple):
    rid: str
    device_indices: torch.Tensor
    node: Any


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
    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
    ) -> int:
        """Check how many tokens are available in external storage.

        Args:
            token_ids: Full token id sequence.
            token_mask: Boolean mask – True for positions to check.
            update_state_for_load: If True, the connector should lock internal
                state so that the query result remains valid until the
                corresponding load is started or cancelled.
            rid: Request id used to track the subsequent load task.

        Returns:
            int: The number of new matched tokens.
        """
        ...

    @abstractmethod
    def cancel_load_task(self, rid: str) -> None:
        """Cancel a previously locked load for the given request.

        Called when GPU memory allocation fails or the load exceeds the
        memory quota.  Releases the lock acquired by
        :meth:`get_new_hit_length` with ``update_state_for_load=True``.

        Args:
            rid: Request id previously passed to :meth:`get_new_hit_length`.
        """
        ...

    @abstractmethod
    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        """Start a batch of load operations from external storage to GPU.

        The connector handles readiness checks, producer allocation, event
        management, and the actual launch internally.

        Args:
            task_id: Caller-assigned task id for completion tracking.
            load_ops: Pending load operations to execute.
        """
        ...

    @abstractmethod
    def check_completed_load_tasks(self) -> List[int]:
        """Check if any load tasks have completed.

        Returns:
            List of load task_ids whose tasks have completed.
        """
        ...

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @abstractmethod
    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:
        """Asynchronously store KV cache for *token_ids* at *kv_indices*.

        Args:
            task_id: Caller-assigned task id for completion tracking.
        """
        ...

    @abstractmethod
    def check_completed_store_tasks(self) -> List[int]:
        """Check if any store tasks have completed.

        Returns:
            List of store task_ids whose tasks have completed.
        """
        ...

    @abstractmethod
    def prefetch(self, rid: str, token_ids: List[int]) -> None:
        """Start prefetching KV cache from external storage for a request.

        Args:
            rid: Request id.
            token_ids: Token id sequence to prefetch.
        """
        ...

    @abstractmethod
    def check_prefetch_completed(self, rid: str) -> bool:
        """Check if the prefetch for the given request has completed.

        Returns:
            True if complete or no prefetch was needed.
        """
        ...

    @abstractmethod
    def cancel_prefetch(self, rid: str) -> None:
        """Cancel an in-progress or pending prefetch for the given request.

        Called when the request is aborted while waiting in the queue.

        Args:
            rid: Request id previously passed to :meth:`prefetch`.
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

    @abstractmethod
    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        """Register the layer transfer counter with the KV cache pool."""
        ...

    def reset(self) -> None:
        """Reset connector state (called on cache reset)."""
        pass

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

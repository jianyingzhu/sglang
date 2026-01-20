import os
import time
import torch
import logging
import threading
import numpy as np
from typing import List, Optional, Dict, Callable

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, RadixKey
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.managers.flexkv_ipc_utils import (
    eventfd,
    eventfd_read,
    eventfd_write,
    send_fds,
    EFD_SEMAPHORE,
)
import socket
import struct

try:
    from flexkv.kvmanager import KVManager
    from flexkv.integration.config import FlexKVConfig
    from flexkv.server.client import KVTPClient
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.common.request import KVResponseStatus
except ImportError as e:
    raise RuntimeError(
        "FlexKV is not installed. Please install it."
    ) from e

logger = logging.getLogger(__name__)


class FlexKVLoadOperation:
    """Represents a pending KV cache load operation from FlexKV to GPU."""
    
    def __init__(
        self,
        task_id: int,                  # FlexKV task_id from get_match (saved to avoid re-query)
        device_indices: torch.Tensor,  # GPU slots for tokens to load
        node_id: int,
        node: Optional[TreeNode] = None,  # Reference to the created TreeNode for cleanup
    ):
        self.task_id = task_id
        self.device_indices = device_indices
        self.node_id = node_id
        self.node = node  # For cleanup when transfer is cancelled
        self.host_hit_length = device_indices.numel()

class FlexKVLayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_event_fds: List[int] = []
        self.load_event_fds = [eventfd(0, EFD_SEMAPHORE) for _ in range(num_layers)] # Use EFD_SEMAPHORE: each read decrements counter by 1
        
        self._finished = True # Track if all layers are done, initially True so that first update_producer() can proceed
        self._last_layer_wait_count = 0  # Track waits on last layer (K and V each wait once)
        # Each event has its own wait_remaining counter (need 2 waits per layer for K and V)
        self.wait_remaining: List[int] = [2] * num_layers

    def reset_for_new_transfer(self):
        """Reset state for a new transfer. Called when producer starts using this event."""
        self._finished = False
        self._last_layer_wait_count = 0
        # Reset wait counters for all layers (need 2 waits per layer for K and V)
        self.wait_remaining = [2] * self._num_layers

    def wait(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        fd = self.load_event_fds[layer_index]
        # logger.debug(f"[FlexKV] eventfd_read START: layer={layer_index}, fd={fd}")
        eventfd_read(fd) # Blocking read - SGLang hangs here until FlexKV signals via eventfd_write
        # logger.debug(f"[FlexKV] eventfd_read DONE: layer={layer_index}, fd={fd}")
        
        # Track waits on last layer and mark finished when all are done
        if layer_index == self._num_layers - 1:
            self._last_layer_wait_count += 1
            if self._last_layer_wait_count >= 2:
                self._finished = True

    # def reset(self):
    #     self._finished = False

    def close(self):
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self):
        self.close()


class FlexKVLayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Extra producer and consumer counters for overlap mode (triple buffering)
        self.num_counters = 3
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        # Wait until count reaches 0 (need 2 waits for K and V, then skip subsequent calls)
        if event.wait_remaining[threshold] <= 0:
            return
        event.wait_remaining[threshold] -= 1
        event.wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1

    def __del__(self):
        # not called
        for event in self.events:
            event.close()
        self.events.clear()


class FlexKVConnector:
    """
    Manages KV cache operations through FlexKV's distributed cache system.
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        page_size: int,
        tp_size: int,
        tp_rank: int,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,        
    ):

        self.flexkv_config = FlexKVConfig.from_env()
        self.flexkv_config.post_init_from_sglang_config(sglang_config=sgl_config,
                                                        tp_size=tp_size,
                                                        page_size=page_size)

        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = tp_rank
        self.k_pool = k_pool
        self.v_pool = v_pool
        self.tp_group = tp_group

        if self.rank == 0:
            self.kv_manager = KVManager(
                model_config=self.flexkv_config.model_config,
                cache_config=self.flexkv_config.cache_config,
                server_recv_port=self.flexkv_config.server_recv_port
            )
            self.kv_manager.start()

        self.tp_client = KVTPClient(self.flexkv_config.gpu_register_port, 0, self.rank)
        self.register_to_server(self.k_pool, self.v_pool)

        # inflight_taskid2reqid: rank 0 only, inflight_reqid2node: all ranks, inflight_skipped_reqids: rank 0 only
        
        # task_id -> req_id mapping of store_async (only maintained on rank 0),
        self.inflight_taskid2reqid: Dict[int, int] = {} if self.rank == 0 else {}
        # req_ids that were skipped (no store launched) on rank 0; used to release locks on all ranks
        self.inflight_skipped_reqids: List[int] = [] if self.rank == 0 else []

        logger.info(f"FlexKV connector initialized for rank {self.rank}")
        
        # Layer-by-layer transfer components
        self.num_layers = sgl_config.num_hidden_layers if sgl_config is not None else 0
        self.layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False
        
        # Socket path for eventfd IPC (from FlexKV config or default)
        self.layerwise_eventfd_socket = os.environ.get(
            'FLEXKV_LAYERWISE_EVENTFD_SOCKET', '/tmp/flexkv_layerwise_eventfd.sock'
        )
        

        self._init_layer_transfer_components()

        if self.rank == 0:
            while not self.kv_manager.is_ready():
                time.sleep(3)
                logger.info("waiting for flexkv to be ready")
            logger.info("flexkv is ready")

    def _init_layer_transfer_components(self):
        """
        Initialize layer-by-layer transfer components for all ranks.
        Each rank creates FlexKVLayerDoneCounter with eventfds and sends them
        to the LayerwiseTransferWorker via Unix domain socket.
        
        The worker receives eventfds from each tp_rank and uses them for 
        layer-by-layer signaling.
        """
        # Create layer done counter with eventfds
        self.layer_done_counter = FlexKVLayerDoneCounter(self.num_layers)
        
        # Connect to FlexKV LayerwiseTransferWorker and send eventfds
        self._send_eventfds_to_worker()
        
        logger.info(f"[FlexKV] Rank {self.rank}: Initialized layer transfer, sent eventfds to worker")

    def _send_eventfds_to_worker(self, max_retries: int = 180, retry_interval: float = 1.0):
        """
        Connect to FlexKV LayerwiseTransferWorker via Unix socket and send all 3 sets of eventfds.
        
        Protocol:
        1. Connect to worker's Unix socket (retry until worker is ready)
        2. Send metadata: tp_rank, tp_size, num_layers, num_counters (as struct)
        3. For each counter (3 sets), send all layer eventfds via send_fds
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        # Retry connection until worker's server socket is ready
        connected = False
        for attempt in range(max_retries):
            try:
                sock.connect(self.layerwise_eventfd_socket)
                logger.info(f"[FlexKV] Rank {self.rank}: Connected to worker socket {self.layerwise_eventfd_socket}")
                connected = True
                break
            except (FileNotFoundError, ConnectionRefusedError) as e:
                if attempt < max_retries - 1:
                    if attempt % 10 == 0:  # Log every 10 attempts to reduce spam
                        logger.info(f"[FlexKV] Rank {self.rank}: Worker not ready, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"[FlexKV] Rank {self.rank}: Failed to connect to worker after {max_retries} attempts")
                    sock.close()
                    raise RuntimeError(f"[FlexKV] Rank {self.rank}: Failed to connect to worker after {max_retries} attempts")
                    self._worker_connected = False
                    return
        
        if not connected:
            sock.close()
            self._worker_connected = False
            return
        
        try:
            # Send metadata: tp_rank, tp_size, num_layers, num_counters
            num_counters = self.layer_done_counter.num_counters
            metadata = struct.pack("iiii", self.rank, self.tp_size, self.num_layers, num_counters)
            sock.sendall(metadata)
            
            # For each counter set (3 sets), send all layer eventfds
            for counter_id in range(num_counters):
                event = self.layer_done_counter.events[counter_id]
                fds = event.load_event_fds  # List of num_layers eventfds
                # Pack counter_id as extra data
                extra_data = struct.pack("i", counter_id)
                send_fds(sock, fds, extra_data)
                logger.debug(f"[FlexKV] Rank {self.rank}: Sent eventfds for counter {counter_id}, num_fds={len(fds)}")
            
            self._worker_connected = True
            logger.info(f"[FlexKV] Rank {self.rank}: Successfully sent all {num_counters} sets of eventfds")
            
        except Exception as e:
            logger.error(f"[FlexKV] Rank {self.rank}: Failed to send eventfds: {e}")
            self._worker_connected = False
        finally:
            sock.close()

    def register_layer_transfer_counter(self, kvcache) -> None:
        """Register layer done counter with the KV cache (only effective on rank 0)."""
        if self.layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self.layer_done_counter)

    def chunk_size(self) -> int:
        """Return the chunk size used by FlexKV."""
        return self.flexkv_config.cache_config.tokens_per_block

    def poll_completed_store_tasks(self) -> List[int]:
        """
        Non-blocking poll to collect completed async store tasks and return their req_ids (rank 0 only).
        Other ranks return an empty list.
        """
        if self.rank == 0:
            inflight_task_ids = list(self.inflight_taskid2reqid.keys())
            if not inflight_task_ids:
                return []
            completed_dict = self.kv_manager.try_wait(task_ids=inflight_task_ids)  # task_id -> KVResponse
            completed_req_ids: List[int] = []
            for task_id in completed_dict:
                req_id = self.inflight_taskid2reqid[task_id]
                del self.inflight_taskid2reqid[task_id]
                completed_req_ids.append(req_id)
            return completed_req_ids
        else:
            return []

    def poll_skipped_store_tasks(self) -> List[int]:
        """
        Non-blocking poll to collect req_ids that were skipped (no store launched).
        Rank 0 returns and clears the list; other ranks return [].
        """
        if self.rank == 0:
            if not self.inflight_skipped_reqids:
                return []
            skipped = self.inflight_skipped_reqids
            self.inflight_skipped_reqids = []
            return skipped
        return []

    def start_load_kv(
        self,
        token_ids: List[int],
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        inflight_reqid2node: Optional[Dict[int, TreeNode]] = None,
        dec_lock_ref_fn: Optional[Callable[[TreeNode], None]] = None,
    ) -> tuple[int, Optional[torch.Tensor]]:
        """
        [Not used now]
        Start loading KV cache from FlexKV storage.

        Args:
            token_ids: List of token IDs to load
            slot_mapping: Tensor mapping for slots
            token_mask: Optional mask indicating which tokens to load from FlexKV

        Returns:
            Tuple of (number of tokens loaded, loaded slot IDs tensor)
        """

        if self.rank == 0:
            token_ids_np = np.array(token_ids, dtype=np.int64) # isinstance(token_ids, list):

            task_id, matched_mask = self.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )

            completed_req_ids = self.poll_completed_store_tasks()
            skipped_req_ids = self.poll_skipped_store_tasks()

            if matched_mask.sum() > 0:
                filtered_slot_mapping = slot_mapping[matched_mask]
                slot_mapping_cpu = filtered_slot_mapping.cpu() if filtered_slot_mapping.is_cuda else filtered_slot_mapping
                self.kv_manager.launch(task_ids=[task_id], slot_mappings=[slot_mapping_cpu])
                response = self.kv_manager.wait([task_id])
                
                if task_id in response and response[task_id].status == KVResponseStatus.SUCCESS:
                    num_loaded = matched_mask.sum().item()
                    # requested_tokens = token_mask.sum().item() if token_mask is not None else len(token_ids)
                    # logger.debug(f"FlexKV loaded {num_loaded}/{requested_tokens} tokens from cache")

                    loaded_slot_ids = filtered_slot_mapping
                    # Update broadcast_data with actual values
                    broadcast_data = {
                        'num_loaded': num_loaded,
                        'loaded_slot_ids': loaded_slot_ids,
                        'completed_req_ids': completed_req_ids,
                        'skipped_req_ids': skipped_req_ids,
                    }
            else:
                broadcast_data = {
                    'num_loaded': 0,
                    'loaded_slot_ids': None,
                    'completed_req_ids': completed_req_ids,
                    'skipped_req_ids': skipped_req_ids,
                }
        else:
            broadcast_data = None

        if self.tp_group is not None and self.tp_size > 1:
            broadcast_data = broadcast_pyobj([broadcast_data], self.rank, self.tp_group, src=0)[0]

            # release locks for completed reqs
            for req_id in broadcast_data['completed_req_ids']:
                if req_id in inflight_reqid2node:
                    node = inflight_reqid2node[req_id]
                    dec_lock_ref_fn(node)
                    del inflight_reqid2node[req_id]
            # release locks for skipped reqs
            for req_id in broadcast_data.get('skipped_req_ids', []):
                if req_id in inflight_reqid2node:
                    node = inflight_reqid2node[req_id]
                    dec_lock_ref_fn(node)
                    del inflight_reqid2node[req_id]

            return broadcast_data['num_loaded'], broadcast_data['loaded_slot_ids']
        else:
            # For single process case, return values from rank 0's broadcast_data
            if self.rank == 0 and broadcast_data is not None:
                # release locks for completed reqs
                for req_id in broadcast_data['completed_req_ids']:
                    if req_id in inflight_reqid2node:
                        node = inflight_reqid2node[req_id]
                        dec_lock_ref_fn(node)
                        del inflight_reqid2node[req_id]
                # release locks for skipped reqs
                for req_id in broadcast_data.get('skipped_req_ids', []):
                    if req_id in inflight_reqid2node:
                        node = inflight_reqid2node[req_id]
                        dec_lock_ref_fn(node)
                        del inflight_reqid2node[req_id]

                return broadcast_data['num_loaded'], broadcast_data['loaded_slot_ids']
            else:
                # Fallback for non-rank0 or when no data was loaded
                return 0, None

    def store_kv_async(self, token_ids: List[int], kv_indices: torch.Tensor, req_id: int) -> int:
        """
        Store KV cache to FlexKV storage asynchronously.

        Args:
            token_ids: List of token IDs to store
            kv_indices: Tensor of KV indices
            req_id: Request ID for tracking

        Returns:
            task_id for tracking the async operation
        """
        try:
            # Only tp0 performs actual FlexKV operations, other ranks return dummy task_id
            if self.rank == 0:
                token_ids_np = np.array(token_ids, dtype=np.int64) # isinstance(token_ids, list)                  
                assert len(token_ids) == len(kv_indices), "token_ids and kv_indices must have the same length"
                
                task_id, unmatched_mask = self.kv_manager.put_match(
                    token_ids=token_ids_np,
                    token_mask=None,  # Store all tokens
                )

                if unmatched_mask.sum() > 0:
                    filtered_kv_indices = kv_indices[unmatched_mask]
                    slot_mapping_cpu = filtered_kv_indices.cpu() if filtered_kv_indices.is_cuda else filtered_kv_indices
                    self.kv_manager.launch(task_ids=[task_id], slot_mappings=[slot_mapping_cpu])
                    self.inflight_taskid2reqid[task_id] = req_id
                    # logger.debug(f"FlexKV storing {unmatched_mask.sum().item()}/{len(token_ids)} tokens to cache (async)")
                    return task_id
                else:
                    # logger.debug(f"All {len(token_ids)} tokens already in FlexKV cache")
                    # record skipped store so locks can be released across ranks
                    self.inflight_skipped_reqids.append(req_id)
                    return -1  # No task launched
            else:
                # Other ranks don't perform actual operations
                return -1

        except Exception as e:
            logger.error(f"FlexKV store_kv_async failed: {e}")
            return -1

    def wait_task(self, task_id: int, timeout: float = 20.0) -> bool:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if task completed successfully, False otherwise
        """
        if task_id < 0:
            return True  # No task to wait for (dummy task or no-op)

        # Only tp0 has real tasks to wait for
        if self.rank == 0:
            try:
                response = self.kv_manager.wait([task_id], timeout=timeout)
                if task_id in response and response[task_id].status == KVResponseStatus.SUCCESS:
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"FlexKV wait_task failed: {e}")
                return False
        else:
            # Other ranks don't have real tasks, so always return success
            return True

    def launch_layerwise_batch_transfer(
        self,
        load_queue: List["FlexKVLoadOperation"],
        producer_id: int = 0,
    ) -> int:
        if self.rank != 0:
            # Other ranks don't perform actual operations
            return 0
        
        task_ids = []
        slot_mappings = []
        
        for op in load_queue:
            # Use saved task_id directly, no need to re-call get_match
            if op.task_id < 0:
                continue
            
            slot_mapping_cpu = op.device_indices.cpu() if op.device_indices.is_cuda else op.device_indices
            task_ids.append(op.task_id)
            slot_mappings.append(slot_mapping_cpu)
        
        if task_ids:
            self.kv_manager.launch(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                as_batch=True,
                layerwise_transfer=True,
                counter_id=producer_id,
            )
        
        return len(task_ids)

    def register_to_server(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]) -> None:
        logger.info("Start register kv_caches")
        assert len(k_caches) == len(v_caches), "k_caches and v_caches must have the same length"
        num_layer = len(k_caches)

        # not mla
        assert k_caches[0].ndim == 3, (
            f"expect kv cached tensor has 3 dim but get shape={k_caches[0].shape}.")

        num_layer = len(k_caches)
        num_blocks = k_caches[0].shape[0]
        num_kv_heads = k_caches[0].shape[1]
        head_size = k_caches[0].shape[2]
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layer,
            num_block=num_blocks,
            tokens_per_block=1,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=False,
            )
        gpu_blocks = k_caches + v_caches
        self.tp_client.register_to_server(gpu_blocks, gpu_layout)
        logger.info("Finish register kv_caches")

    def shutdown(self) -> None:
        """Shutdown FlexKV connection."""
        self.kv_manager.shutdown()


class FlexKVRadixCache(RadixCache):
    def __init__(
        self,
        params,
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        # Initialize attributes needed by reset() method before calling super().__init__()
        self.rank = rank
        self.tp_group = tp_group
        
        # Track req_id -> gpu last node, for all ranks
        self.inflight_reqid2node: Dict[int, TreeNode] = {}
        self.sts_total_seq_len = 0
        self.sts_gpu_cache_len = 0
        self.sts_flexkv_cache_len = 0

        # Initialize FlexKV connector before super().__init__() since reset() method may use it
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.flexkv_connector = FlexKVConnector(
            sgl_config=model_config,
            page_size=params.page_size,
            tp_size=tp_size,
            tp_rank=rank,
            tp_group=tp_group,            
            k_pool=getattr(
                kvcache,
                "k_buffer",
                getattr(params.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            ),
            v_pool=getattr(
                kvcache,
                "v_buffer",
                getattr(params.token_to_kv_pool_allocator._kvcache, "v_buffer"),
            ),
        )
        
        # Get layer transfer components from connector (created on rank 0 only)
        self.layer_done_counter = self.flexkv_connector.layer_done_counter
        self.flexkv_connector.register_layer_transfer_counter(kvcache)

        self.num_layers = model_config.num_hidden_layers if model_config is not None else 0
        self.tp_size = tp_size

        # Load queue for pending FlexKV -> GPU transfer operations
        self.load_queue: List[FlexKVLoadOperation] = []
        # Pending load info from match_prefix: rid -> (task_id, key, gpu_cached_len)
        # Use dict with rid as key to handle out-of-order or cancelled requests
        self.pending_load_info: Dict[str, tuple] = {}
        # Track ongoing load operations: node_id -> (node, producer_id)
        self.ongoing_load_back: Dict[int, tuple] = {}
        
        super().__init__(params)

    def reset(self):
        super().reset()
        # Wait for all in-flight tasks before reset
        # Only rank 0 needs to wait, other ranks just clear their mappings
        if self.rank == 0:
            task_ids = list(self.flexkv_connector.inflight_taskid2reqid.keys())
            for task_id in task_ids:
                self.flexkv_connector.wait_task(task_id)
        
        # All ranks clean up their node mappings
        for _, node in self.inflight_reqid2node.items():
            self.dec_lock_ref(node)
        self.inflight_reqid2node.clear()
        
        for _, (node, _) in self.ongoing_load_back.items():
            self.dec_lock_ref(node)
        self.ongoing_load_back.clear()

        self.load_queue.clear()
        self.pending_load_info.clear()       

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        # self.sts_total_seq_len += len(key)
        # if self.disable or not key:
        #     return super().match_prefix(key, **kwargs)

        # if self.page_size != 1:
        #     aligned_len = len(key) // self.page_size * self.page_size
        #     key = key[:aligned_len]

        # base_res = super().match_prefix(key, **kwargs)
        # value: torch.Tensor = base_res.device_indices
        # self.sts_gpu_cache_len += value.numel()
        
        # last_node: TreeNode = base_res.last_device_node

        # uncached_len = len(key) - value.numel()
        # if uncached_len == 0:
        #     return base_res

        # chunk_size = self.flexkv_connector.chunk_size()
        # prefix_pad = value.numel() % chunk_size

        # if self.token_to_kv_pool_allocator.available_size() < uncached_len:
        #     self.evict(uncached_len)

        # token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        # if token_slots is None:
        #     return base_res

        # slot_mapping = torch.cat(
        #     [
        #         torch.full((value.numel(),), -1, dtype=torch.int64, device=self.device),
        #         token_slots.detach().clone().to(torch.int64).to(self.device),
        #     ]
        # )

        # token_mask = torch.zeros(len(key), dtype=torch.bool)
        # token_mask[value.numel():] = True  # Only load uncached tokens

        # num_retrieved, loaded_slot_ids = self.flexkv_connector.start_load_kv(
        #     token_ids=key.token_ids,
        #     slot_mapping=slot_mapping,
        #     token_mask=token_mask,
        #     inflight_reqid2node=self.inflight_reqid2node,
        #     dec_lock_ref_fn=self.dec_lock_ref,
        # )
        # self.sts_flexkv_cache_len += num_retrieved
        # if self.rank == 0:
        #     # logger.debug("num_retrieved_tokens: %s", num_retrieved)
        #     logger.info(f"[FlexKV stats] total_seq_len={self.sts_total_seq_len}, "
        #     f"gpu_cache_len={self.sts_gpu_cache_len}, flexkv_cache_len={self.sts_flexkv_cache_len}, "
        #     f"gpu_cache_ratio={self.sts_gpu_cache_len / self.sts_total_seq_len}, "
        #     f"flexkv_cache_ratio={self.sts_flexkv_cache_len / self.sts_total_seq_len}")

        # if num_retrieved > prefix_pad:
        #     fetched = num_retrieved - prefix_pad
        #     # Free unused token slots
        #     self.token_to_kv_pool_allocator.free(
        #         token_slots[fetched:]
        #     )
            
        #     new_node = TreeNode()
        #     start = value.numel()
        #     end = start + fetched
        #     new_node.key = key[start:end]
        #     new_node.value = token_slots[:fetched]
        #     new_node.parent = last_node
        #     last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        #     last_node = new_node

        #     value = torch.cat([value, token_slots[:fetched]])
        #     self.evictable_size_ += fetched

        #     self._record_store_event(new_node.parent)
        #     self._record_store_event(new_node)

        #     return MatchResult(
        #         device_indices=value,
        #         last_device_node=last_node,
        #         last_host_node=last_node,
        #     )
        # else:
        #     logger.debug(f"FlexKV retrieved {num_retrieved} tokens but need at least {prefix_pad} for alignment, freeing all allocated slots")
        #     self.token_to_kv_pool_allocator.free(token_slots)

        # return base_res

        # ===== match only, no transfer =====
        self.sts_total_seq_len += len(key)
        if self.disable or not key:
            return super().match_prefix(key, **kwargs)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        # First, match against GPU radix cache
        base_res = super().match_prefix(key, **kwargs)
        value: torch.Tensor = base_res.device_indices
        self.sts_gpu_cache_len += value.numel()

        last_node: TreeNode = base_res.last_device_node

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        # Query FlexKV to see what tokens are available (match only, no transfer)
        flexkv_hit_length = 0
        flexkv_task_id = -1
        if self.rank == 0:
            token_ids_np = np.array(key.token_ids, dtype=np.int64)
            token_mask = torch.zeros(len(key), dtype=torch.bool)
            token_mask[value.numel():] = True  # Only check uncached tokens

            flexkv_task_id, matched_mask = self.flexkv_connector.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )
            # not launch task of transfer, just match
            flexkv_hit_length = int(matched_mask.sum()) if matched_mask is not None else 0

        # Broadcast flexkv_hit_length and task_id to all ranks if in TP mode
        if self.tp_group is not None and self.flexkv_connector.tp_size > 1:
            broadcast_data = broadcast_pyobj([{
                'flexkv_hit_length': flexkv_hit_length,
                'flexkv_task_id': flexkv_task_id,
            }], self.rank, self.tp_group, src=0)[0]
            flexkv_hit_length = broadcast_data['flexkv_hit_length']
            flexkv_task_id = broadcast_data['flexkv_task_id']

        self.sts_flexkv_cache_len += flexkv_hit_length

        # Store (task_id, key, gpu_cached_len) for init_load_back (avoid re-calling get_match)
        # Use rid as key to support out-of-order or cancelled requests
        rid = kwargs.get('rid')
        if rid is None:
            raise ValueError("rid is required for FlexKV match_prefix")
        if flexkv_hit_length > 0:
            self.pending_load_info[rid] = (flexkv_task_id, key, value.numel())

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node, # not used for FlexKV, used for hicache prefetch and init_load_back
            host_hit_length=flexkv_hit_length,
        )

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
        rid: Optional[str] = None,
    ):
        """
        Allocate GPU memory, create new TreeNode, and add the load operation to load_queue.
        The actual transfer is triggered by ready_to_load_host_cache().
        
        Args:
            last_node: The last node from match_prefix
            host_hit_length: Number of tokens hit in FlexKV storage
            mem_quota: Optional memory quota limit
            rid: Request ID to look up pending load info
            
        Returns:
            Tuple of (device_indices tensor, updated last_node)
        """
        if host_hit_length <= 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        # Get (task_id, key, gpu_cached_len) stored during match_prefix using rid as key
        if rid is None:
            raise ValueError("rid is required for FlexKV init_load_back")

        if rid not in self.pending_load_info:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        task_id, key, gpu_cached_len = self.pending_load_info.pop(rid)
        
        # Check memory quota
        if mem_quota is not None and host_hit_length > mem_quota:
            logger.debug(f"[FlexKV] host_hit_length {host_hit_length} exceeds mem_quota {mem_quota}, skipping load")
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        # Allocate GPU memory for the tokens to load
        device_indices = self.token_to_kv_pool_allocator.alloc(host_hit_length)
        if device_indices is None:
            # Try eviction and retry
            self.evict(host_hit_length)
            device_indices = self.token_to_kv_pool_allocator.alloc(host_hit_length)
        
        if device_indices is None:
            logger.warning(f"[FlexKV] Failed to allocate {host_hit_length} GPU slots for load")
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        # ===== Create new TreeNode after alloc =====
        new_node = TreeNode()
        start = gpu_cached_len
        end = start + host_hit_length
        new_node.key = key[start:end]
        new_node.value = device_indices
        new_node.parent = last_node
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        
        # Update evictable_size
        self.evictable_size_ += len(device_indices)
        
        # Lock new node to prevent eviction during transfer
        self.inc_lock_ref(new_node)
        
        # Create load operation with saved task_id (no need to re-call get_match)
        load_op = FlexKVLoadOperation(
            task_id=task_id,
            device_indices=device_indices,
            node_id=new_node.id,  # Use new node's id
            node=new_node,  # Keep reference for cleanup
        )
        self.load_queue.append(load_op)
        
        return device_indices, new_node
    
    def ready_to_load_host_cache(self) -> int:
        """
        Trigger the actual layer-by-layer transfer from FlexKV to GPU.
        The LayerwiseTransferWorker in FlexKV performs the transfer and signals
        each layer completion via eventfd_write.

        Flow:
        1. SGLang creates eventfds in FlexKVLayerLoadingEvent (during init)
        2. SGLang sends those eventfds to the worker via Unix socket (during init)
        3. SGLang triggers transfer via KVManager.launch() 
        4. Worker receives task and performs layer-by-layer transfer
        5. Worker signals each layer completion via eventfd_write
        6. SGLang waits on each layer's eventfd before computing

        Returns:
            Consumer index for the scheduler to track layer-by-layer progress,
            or -1 if no operations to process.
        """
        if not self.flexkv_connector._worker_connected:
            raise RuntimeError("[FlexKV] Worker not connected, skipping layer-by-layer transfer")
        
        if self.layer_done_counter is None:
            raise RuntimeError("[FlexKV] Layer done counter not available, skipping")
            self.load_queue.clear()
            return -1
        
        if not self.load_queue:
            return -1
        
        producer_id = self.layer_done_counter.update_producer()
        # Reset event state for new transfer (marks as not finished and resets wait counter)
        self.layer_done_counter.events[producer_id].reset_for_new_transfer()
        
        # Track nodes being loaded with their producer_id for later unlock
        for op in self.load_queue:
            if op.node is not None:
                self.ongoing_load_back[op.node_id] = (op.node, producer_id)
        
        # Launch layerwise batch transfer (rank 0 only, other ranks do nothing)
        # Pass producer_id so FlexKV knows which eventfd set to use for notification
        self.flexkv_connector.launch_layerwise_batch_transfer(self.load_queue, producer_id)
        
        self.load_queue.clear()
        
        return producer_id

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        super().cache_finished_req(req, is_insert=is_insert)

        if req.req_pool_idx is None:
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_last_node = req.last_node
        if new_last_node is None:
            return

        self.inc_lock_ref(new_last_node)
        try:
            task_id = self.flexkv_connector.store_kv_async(
                token_ids=token_ids,
                kv_indices=kv_indices,
                req_id=req.req_pool_idx,
            )
        except Exception as e:
            logger.error(f"[FlexKV] Failed to store KV: {e}")
            return

        if req.req_pool_idx in self.inflight_reqid2node:
            self.dec_lock_ref(self.inflight_reqid2node[req.req_pool_idx])
            del self.inflight_reqid2node[req.req_pool_idx]

        self.inflight_reqid2node[req.req_pool_idx] = new_last_node

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        new_prefix_len = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key), page_aligned_kv_indices, chunked=chunked
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        match_res = super().match_prefix(RadixKey(token_ids=page_aligned_token_ids, extra_key=req.extra_key))
        new_indices = match_res.device_indices
        new_last_node = match_res.last_device_node
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

        self.inc_lock_ref(new_last_node)
        try:
            task_id = self.flexkv_connector.store_kv_async(
                token_ids=page_aligned_token_ids,
                kv_indices=page_aligned_kv_indices,
                req_id=req.req_pool_idx,
            )
        except Exception as e:
            logger.error(f"[FlexKV] Failed to store KV: {e}")
            return

        if req.req_pool_idx in self.inflight_reqid2node:
            self.dec_lock_ref(self.inflight_reqid2node[req.req_pool_idx])
            del self.inflight_reqid2node[req.req_pool_idx]

        self.inflight_reqid2node[req.req_pool_idx] = new_last_node

    def evict(self, num_tokens: int) -> None:
        """
        Try non-blocking release of completed FlexKV store tasks, evict, and only
        if insufficient tokens are freed, block-wait remaining tasks and evict again.
        """
        if self.disable:
            return

        # Step 1: Non-blocking poll to release completed/skimmed store locks
        try:
            self.writing_check()
        except Exception:
            pass

        # Step 2: Attempt eviction with currently evictable nodes
        evicted = super().evict(num_tokens)
        if evicted >= num_tokens:
            return

        # Step 3: Not enough freed. Block-wait remaining FlexKV tasks, then release all locks and evict the rest.
        remaining_reqids = list(self.inflight_reqid2node.keys())

        if self.flexkv_connector.rank == 0:
            task_ids = list(self.flexkv_connector.inflight_taskid2reqid.keys())
            for task_id in task_ids:
                self.flexkv_connector.wait_task(task_id)
            self.flexkv_connector.inflight_taskid2reqid.clear()
            
        for req_id in remaining_reqids:
            node = self.inflight_reqid2node[req_id]
            self.dec_lock_ref(node)
        self.inflight_reqid2node.clear()

        remaining_to_evict = num_tokens - evicted
        if remaining_to_evict > 0:
            super().evict(remaining_to_evict)
    
    def pretty_print(self):
        super().pretty_print()
        logger.debug(
            "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
        )

    def loading_check(self):
        """
        Check for completed load operations and release corresponding locks.
        """
        if self.layer_done_counter is None:
            return
        
        if len(self.ongoing_load_back) == 0:
            return
        
        completed_nodes = []
        for node_id, (node, producer_id) in list(self.ongoing_load_back.items()):
            # Check if the loading event for this producer_id has finished
            event = self.layer_done_counter.events[producer_id]
            if event._finished:
                completed_nodes.append(node_id)
        
        for node_id in completed_nodes:
            node, producer_id = self.ongoing_load_back.pop(node_id)
            self.dec_lock_ref(node)
        #     logger.debug(
        #         f"[FlexKV] loading_check: released lock for node {node_id}, "
        #         f"producer_id={producer_id}, key_len={len(node.key)}"
        #     )
        
        # if completed_nodes:
        #     logger.debug(
        #         f"[FlexKV] loading_check: released {len(completed_nodes)} load locks, "
        #         f"protected_size={self.protected_size()} evictable_size={self.evictable_size()}"
        #     )

    def check_kv_events(self):
        self.writing_check()
        self.loading_check()
        # Log FlexKV stats periodically
        if self.rank == 0 and self.sts_total_seq_len > 0:
            logger.info(f"[FlexKV stats] total_seq_len={self.sts_total_seq_len}, "
                        f"gpu_cache_len={self.sts_gpu_cache_len}, flexkv_cache_len={self.sts_flexkv_cache_len}, "
                        f"gpu_cache_ratio={self.sts_gpu_cache_len / self.sts_total_seq_len:.4f}, "
                        f"flexkv_cache_ratio={self.sts_flexkv_cache_len / self.sts_total_seq_len:.4f}")

    def writing_check(self) -> None:
        """
        Poll for completed store tasks and release corresponding locks.
        Works for both single-rank and tensor-parallel multi-rank cases.
        """
        # Rank 0 polls the FlexKV manager; others will receive the broadcast payload.
        completed_req_ids: List[int]
        if self.flexkv_connector.rank == 0:
            completed_req_ids = self.flexkv_connector.poll_completed_store_tasks()
            skipped_req_ids = self.flexkv_connector.poll_skipped_store_tasks()
        else:
            completed_req_ids = []
            skipped_req_ids = []

        # Broadcast to all ranks if using tensor-parallel group
        if self.tp_group is not None and self.flexkv_connector.tp_size > 1:
            payload = {'completed_req_ids': completed_req_ids, 'skipped_req_ids': skipped_req_ids} if self.rank == 0 else None
            payload = broadcast_pyobj([payload], self.rank, self.tp_group, src=0)[0]
            to_release = (payload.get('completed_req_ids') or []) + (payload.get('skipped_req_ids') or [])
        else:
            to_release = completed_req_ids + skipped_req_ids

        released = 0
        for req_id in to_release:
            if req_id in self.inflight_reqid2node:
                node = self.inflight_reqid2node[req_id]
                # logger.debug(
                #     f"[FlexKV] periodic release store_lock req={req_id} "
                #     f"node={node.id} key_len={len(node.key)}"
                # )
                self.dec_lock_ref(node)
                del self.inflight_reqid2node[req_id]
                released += 1
        # if released:
        #     logger.debug(
        #         f"[FlexKV] periodic released {released} store locks; "
        #         f"protected_size={self.protected_size()} evictable_size={self.evictable_size()}"
        #     )


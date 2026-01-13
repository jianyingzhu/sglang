import os
import time
import torch
import logging
import ctypes
import threading
import socket
import struct
from typing import Optional, List, Dict

from sglang.srt.managers.flexkv_ipc_utils import (
    eventfd,
    eventfd_write,
    EFD_SEMAPHORE,
    cudart,
    recv_fds,
)

logger = logging.getLogger(__name__)


# Default socket path for IPC
DEFAULT_SOCKET_PATH = "/tmp/flexkv_simulated_worker.sock"

# Message types
MSG_TYPE_INIT_FDS = 1      # Send all eventfds at init
MSG_TYPE_START_TRANSFER = 2  # Notify to start transfer with specific producer_id


class FlexKVSimulatedWorker:
    def __init__(
        self,
        num_layers: int,
        tp_size: int = 1,
        gpu_id: int = 0,
        socket_path: str = DEFAULT_SOCKET_PATH,
        num_counters: int = 3,  # Triple buffering by default
    ):
        self.num_layers = num_layers
        self.tp_size = tp_size
        self.gpu_id = gpu_id
        self.socket_path = socket_path
        self.num_counters = num_counters
        
        # All eventfds received at init: producer_id -> List[eventfd]
        self.all_event_fds: Dict[int, List[int]] = {}
        
        # Current batch's eventfds (set by start_transfer command)
        self.current_event_fds: List[int] = []
        self.current_producer_id: int = -1
        
        # Server socket for receiving eventfds
        self._server_socket: Optional[socket.socket] = None
        self._client_conn: Optional[socket.socket] = None
        
        # Client socket for sending eventfds (SGLang side)
        self._client_socket: Optional[socket.socket] = None
        
        # Initialize CUDA resources
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None
        
        # Keep references to callbacks to prevent GC
        self._callbacks: List[ctypes.CFUNCTYPE] = []
        
        # Track if transfer is in progress
        self._transfer_in_progress = False
        
        # Server listening thread
        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_flag = False
        
        # Flag to indicate if all eventfds have been received
        self._fds_initialized = False
        
        logger.debug(f"[FlexKVSimulatedWorker] Initialized with {num_layers} layers, "
                    f"socket_path={socket_path}, num_counters={num_counters}")

    def set_event_fds(self, all_event_fds: Dict[int, List[int]]) -> None:
        """
        Directly set eventfds for same-process mode (no socket communication needed).
        
        This is used when FlexKVSimulatedWorker runs in the same process as SGLang,
        allowing direct sharing of eventfds without IPC.
        
        Args:
            all_event_fds: Dict mapping counter_id (producer_id) to list of eventfds per layer
        """
        self.all_event_fds = all_event_fds
        self._fds_initialized = True
        logger.info(f"[FlexKVSimulatedWorker] Set {len(all_event_fds)} counters with eventfds directly (same-process mode)")

    def accept_connection(self, timeout: float = 30.0) -> bool:
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._shutdown_flag:
                return False
            try:
                self._client_conn, addr = self._server_socket.accept()
                logger.info(f"[FlexKVSimulatedWorker] Accepted connection from SGLang")
                return True
            except socket.timeout:
                continue
        
        logger.warning("[FlexKVSimulatedWorker] Timeout waiting for connection")
        return False

    def receive_all_event_fds(self, wait_for_transfer: bool = False) -> int:
        """
        Receive all eventfds from SGLang at initialization, optionally wait for start transfer command.
        
        FlexKV only needs to receive the eventfds that SGLang passes over. SGLang creates and
        owns the eventfds, FlexKV just receives them and uses them to signal layer completion.
        
        Protocol:
        - Header: msg_type(1) + num_counters(1) + num_layers(2) = 4 bytes
        - For each counter: receive num_layers eventfds
        - If wait_for_transfer=True, also wait for start_transfer command (msg_type(1) + producer_id(1))
        
        Args:
            wait_for_transfer: If True, also wait for and process start_transfer command
            
        Returns:
            producer_id if wait_for_transfer=True and received successfully,
            0 if only eventfds received successfully,
            -1 on error
        """
        if self._client_conn is None:
            logger.error("[FlexKVSimulatedWorker] No client connection")
            return -1
        
        try:
            # Receive all fds for all counters at once
            total_fds = self.num_counters * self.num_layers
            fds, data = recv_fds(self._client_conn, total_fds)
            
            # Parse header
            if len(data) < 4:
                logger.error(f"[FlexKVSimulatedWorker] Invalid header length: {len(data)}")
                return -1
            
            msg_type, num_counters, num_layers = struct.unpack("BBH", data[:4])
            
            if msg_type != MSG_TYPE_INIT_FDS:
                logger.error(f"[FlexKVSimulatedWorker] Expected MSG_TYPE_INIT_FDS, got {msg_type}")
                return -1
            
            if num_counters != self.num_counters or num_layers != self.num_layers:
                logger.error(f"[FlexKVSimulatedWorker] Mismatch: expected {self.num_counters}x{self.num_layers}, "
                           f"got {num_counters}x{num_layers}")
                return -1
            
            if len(fds) != total_fds:
                logger.error(f"[FlexKVSimulatedWorker] Expected {total_fds} fds, got {len(fds)}")
                return -1
            
            # Distribute fds to each counter
            for counter_id in range(num_counters):
                start_idx = counter_id * num_layers
                end_idx = start_idx + num_layers
                self.all_event_fds[counter_id] = fds[start_idx:end_idx]
            
            self._fds_initialized = True
            logger.info(f"[FlexKVSimulatedWorker] Received {total_fds} eventfds for {num_counters} counters")
            
            # Optionally wait for start_transfer command
            if wait_for_transfer:
                producer_id = self.wait_for_transfer_command()
                return producer_id
            
            return 0
            
        except Exception as e:
            logger.error(f"[FlexKVSimulatedWorker] Error receiving all eventfds: {e}")
            return -1

    def transfer_all_layers(self, producer_id: int = -1) -> bool:
        if producer_id >= 0:
            self.current_producer_id = producer_id
            if producer_id in self.all_event_fds:
                self.current_event_fds = self.all_event_fds[producer_id]
            else:
                logger.error(f"[FlexKVSimulatedWorker] No eventfds for producer_id {producer_id}")
                return False
        
        if not self.current_event_fds:
            logger.warning("[FlexKVSimulatedWorker] No eventfds to signal")
            return False
            
        if self._transfer_in_progress:
            logger.warning("[FlexKVSimulatedWorker] Transfer already in progress")
            return False
            
        self._transfer_in_progress = True
        logger.info(f"[FlexKVSimulatedWorker] Starting transfer for {self.num_layers} layers, "
                   f"producer_id={self.current_producer_id}")
        
        for layer_id in range(self.num_layers):
            self.transfer_layer(layer_id)
            
        logger.debug("[FlexKVSimulatedWorker] All layer transfers scheduled")
        return True

    def transfer_layer(self, layer_id: int):
        if layer_id >= self.num_layers:
            raise ValueError(f"Invalid layer_id {layer_id}, max {self.num_layers - 1}")
        
        if layer_id >= len(self.current_event_fds):
            logger.error(f"[FlexKVSimulatedWorker] No eventfd for layer {layer_id}")
            return
        
        # 1. Disk to Host (CPU Operation) - simulated
        self._disk_to_host_op(layer_id)
        
        # 2. Host to Device (GPU Kernel) & Callback
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                self._host_to_device_kernel(layer_id)
                
                # 3. Insert Host Callback into the stream
                if cudart:
                    self._schedule_callback(layer_id)
                else:
                    # Fallback: directly signal (not ideal but works for testing)
                    logger.warning(f"[FlexKVSimulatedWorker] cudart not found, using sync fallback for layer {layer_id}")
                    self.transfer_stream.synchronize()
                    eventfd_write(self.current_event_fds[layer_id], self.tp_size)
        else:
            # No CUDA available, directly signal
            eventfd_write(self.current_event_fds[layer_id], self.tp_size)

    def _schedule_callback(self, layer_id: int):
        """Schedule a CUDA host callback to signal layer completion via eventfd."""
        event_fd = self.current_event_fds[layer_id]
        tp_size = self.tp_size
        num_layers = self.num_layers
        
        def callback_func(user_data):
            try:
                eventfd_write(event_fd, tp_size)
                logger.debug(f"[FlexKVSimulatedWorker] Layer {layer_id} callback: signaled via eventfd")
            except Exception as e:
                logger.error(f"[FlexKVSimulatedWorker] Error in CUDA callback for layer {layer_id}: {e}")
            
            # Mark transfer as complete when last layer signals
            if layer_id == num_layers - 1:
                self._transfer_in_progress = False

        c_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(callback_func)
        self._callbacks.append(c_callback)
        
        # Get stream pointer
        stream_ptr = self.transfer_stream.cuda_stream
        
        # Launch the host function callback
        err = cudart.cudaLaunchHostFunc(
            ctypes.c_void_p(stream_ptr),
            c_callback,
            None
        )
        if err != 0:
            logger.error(f"[FlexKVSimulatedWorker] cudaLaunchHostFunc failed with error {err} for layer {layer_id}")

    def _disk_to_host_op(self, layer_id: int):
        """Simulate disk to host transfer (CPU operation)."""
        # In real implementation, this would read from disk/SSD
        # Small delay to simulate I/O latency
        time.sleep(0.001)

    def _host_to_device_kernel(self, layer_id: int):
        """Simulate host to device transfer (GPU kernel)."""
        # In real implementation, this would be cudaMemcpyAsync
        # Here we just do a simple GPU operation to simulate work
        t = torch.zeros(1024, device="cuda", dtype=torch.float32)
        t.add_(1.0)

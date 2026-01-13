import os
import socket
import threading
import time
import torch
import logging
import struct
import ctypes
from typing import Optional, List

from sglang.srt.managers.flexkv_ipc_utils import send_fds, eventfd, eventfd_write, EFD_SEMAPHORE, cudart

logger = logging.getLogger(__name__)

class FlexKVWorker:
    """
    Simulates the FlexKV process that handles hierarchical KV cache transfer.
    
    Flow:
    1. Disk -> Host (CPU Operation)
    2. Host -> Device (GPU Kernel)
    3. Signal SGLang via eventfd (triggered by CUDA Host Callback)
    
    Each layer has its own eventfd for synchronization.
    """
    def __init__(self, socket_path: str, num_layers: int, gpu_id: int = 0, tp_size: int = 1):
        self.socket_path = socket_path
        self.num_layers = num_layers
        self.gpu_id = gpu_id
        self.tp_size = tp_size
        self.running = True
        self.server_sock: Optional[socket.socket] = None
        
        # Create one eventfd per layer
        self.event_fds: List[int] = []
        for _ in range(num_layers):
            # Use EFD_SEMAPHORE so that each read decrements the counter by 1
            self.event_fds.append(eventfd(0, EFD_SEMAPHORE))
        
        logger.info(f"Created {num_layers} eventfds for layer synchronization.")

        # Initialize CUDA resources
        torch.cuda.set_device(self.gpu_id)
        self.transfer_stream = torch.cuda.Stream()
        
        # Keep references to callbacks to prevent GC
        self._callbacks = [] 
        
        # Thread for handling connection
        self.conn_thread = threading.Thread(target=self._accept_conn, daemon=True)

    def start(self):
        # Clean up old socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(1)
        
        self.conn_thread.start()
        logger.info(f"FlexKVWorker started on {self.socket_path}")

    def _accept_conn(self):
        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                logger.info("Accepted connection from SGLang")
                
                # Protocol: send num_layers (4 bytes) followed by all eventfds
                data = struct.pack("!I", self.num_layers)
                send_fds(conn, self.event_fds, data)
                conn.close()
            except OSError:
                break

    def transfer_layer(self, layer_id: int):
        """
        Perform the transfer for a specific layer.
        This function returns immediately after scheduling the work.
        """
        if layer_id >= self.num_layers:
            raise ValueError(f"Invalid layer_id {layer_id}, max {self.num_layers - 1}")
        
        # 1. Disk to Host (CPU Operation)
        self._disk_to_host_op(layer_id)
        
        # 2. Host to Device (GPU Kernel) & Callback
        with torch.cuda.stream(self.transfer_stream):
            self._host_to_device_kernel(layer_id)
            
            # 3. Insert Host Callback into the stream
            if cudart:
                self._schedule_callback(layer_id)
            else:
                print(f"cudart not found")
                raise RuntimeError("cudart not found")

    def _schedule_callback(self, layer_id):
        def callback_func(user_data):
            try:
                eventfd_write(self.event_fds[layer_id], self.tp_size) # write to eventfd to signal SGLang
            except Exception as e:
                print(f"Error in CUDA callback: {e}")

        c_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(callback_func)
        self._callbacks.append(c_callback)
        
        # Get stream pointer
        stream_ptr = self.transfer_stream.cuda_stream
        
        # Launch
        err = cudart.cudaLaunchHostFunc(
            ctypes.c_void_p(stream_ptr),
            c_callback,
            None
        )
        if err != 0:
            logger.error(f"cudaLaunchHostFunc failed with error {err}")

    def _disk_to_host_op(self, layer_id):
        time.sleep(0.5) 

    def _host_to_device_kernel(self, layer_id):
        t = torch.zeros(4096 * 4096, device="cuda", dtype=torch.float32)
        t.add_(1.0)
    
    def shutdown(self):
        self.running = False
        if self.server_sock:
            self.server_sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Close all eventfds
        for fd in self.event_fds:
            os.close(fd)
        self.event_fds.clear()

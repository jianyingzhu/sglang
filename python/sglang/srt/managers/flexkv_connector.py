import os
import socket
import logging
import struct
import time
from typing import Optional, List
from sglang.srt.managers.flexkv_ipc_utils import eventfd_read

logger = logging.getLogger(__name__)

class FlexKVConnector:
    """
    Connector for FlexKV to handle hierarchical KV cache transfer synchronization.
    
    It uses a Unix domain socket to receive eventfds from the FlexKV process.
    Each layer has its own eventfd used to signal when that layer's KV cache 
    transfer (disk -> host -> device) is complete.
    """
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.event_fds: List[int] = []
        self.sock: Optional[socket.socket] = None
        self.num_layers = 0
        self.connect()

    def connect(self):
        """Connect to FlexKV process and receive the eventfds."""
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Wait for FlexKV socket to be ready (retry logic)
            for _ in range(10):
                try:
                    self.sock.connect(self.socket_path)
                    break
                except FileNotFoundError:
                    time.sleep(0.1)
            else:
                # Try one last time or fail
                self.sock.connect(self.socket_path)
            
            # Receive data and fds together via recvmsg
            # Protocol: num_layers (4 bytes int) as data, eventfds as ancillary
            data, fds = self._recv_fds_with_data()
            
            self.num_layers = struct.unpack("!I", data[:4])[0]
            self.event_fds = fds
            
            if len(self.event_fds) != self.num_layers:
                raise RuntimeError(f"Expected {self.num_layers} eventfds, got {len(self.event_fds)}")
            
            logger.info(f"Connected to FlexKV. num_layers={self.num_layers}, received {len(self.event_fds)} eventfds")
            
            self.sock.close()
            self.sock = None
        except Exception as e:
            logger.error(f"Failed to connect to FlexKV: {e}")
            raise

    def _recv_fds_with_data(self):
        """Receive data and multiple fds via Unix domain socket."""
        # We need enough ancillary buffer for potentially many fds
        # Assume max 256 layers for buffer sizing
        max_fds = 256
        anc_buf_size = socket.CMSG_SPACE(max_fds * struct.calcsize("i"))
        
        data, ancdata, flags, addr = self.sock.recvmsg(256, anc_buf_size)
        
        fds = []
        for level, ctype, cdata in ancdata:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                num_received = len(cdata) // struct.calcsize("i")
                fds = list(struct.unpack(f"{num_received}i", cdata[:num_received * struct.calcsize("i")]))
                break
        
        if not fds:
            raise RuntimeError("did not receive fds via SCM_RIGHTS")
        
        return data, fds

    def wait_for_layer_transfer(self, layer_id: int):
        """
        Block until the KV cache for the specified layer is transferred to GPU.
        
        This should be called before executing the forward pass of a layer
        that requires the transferred KV cache.
        """
        if not self.event_fds:
            raise RuntimeError("FlexKV connector not connected")
        
        if layer_id >= self.num_layers:
            raise ValueError(f"Invalid layer_id {layer_id}, max {self.num_layers - 1}")

        # Blocking read on the layer's eventfd
        _ = eventfd_read(self.event_fds[layer_id])
           
    def close(self):
        # Close all eventfds
        for fd in self.event_fds:
            os.close(fd)
        self.event_fds.clear()

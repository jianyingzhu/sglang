import time
import threading
import torch
import os
from sglang.srt.managers.flexkv_worker import FlexKVWorker
from sglang.srt.managers.flexkv_connector import FlexKVConnector

SOCK_PATH = "/tmp/flexkv_test.sock"

def sglang_simulation(num_layers):
    """Simulates the SGLang inference loop."""
    connector = FlexKVConnector(SOCK_PATH)
    print("[SGLang] Connected to FlexKV")

    for i in range(num_layers):
        print(f"[SGLang] Before layer {i} computation. Waiting for KV cache...")
        
        # This blocks until FlexKV signals
        start_wait = time.time()
        connector.wait_for_layer_transfer(i)
        end_wait = time.time()
        
        print(f"[SGLang] Layer {i} KV cache ready! Waited {end_wait - start_wait:.4f}s. Computing layer {i}...")
        
        # Simulate computation of layer
        torch.matmul(torch.randn(1024, 1024), torch.randn(1024, 1024))
        
    print("[SGLang] Inference complete.")
    connector.close()

def flexkv_simulation(worker, num_layers):
    """Simulates the FlexKV transfer loop."""
    print("[FlexKV] Starting transfers...")
    for i in range(num_layers):
        print(f"[FlexKV] Start transferring layer {i}...")
        worker.transfer_layer(i)
        print(f"[FlexKV] Layer {i} transferred")
        # In reality, FlexKV might be faster or slower than SGLang, 
        # or prefetching ahead. Here we just trigger them.
        # time.sleep(0.02) # Simulate gap between transfers
    
def main():
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
        
    num_layers = 5
    # 1. Start FlexKV Worker
    worker = FlexKVWorker(SOCK_PATH, num_layers=num_layers)
    worker.start()
    
    # 2. Start SGLang simulation (in a separate thread)
    sglang_thread = threading.Thread(target=sglang_simulation, args=(num_layers,))
    sglang_thread.start()
    
    # Wait a bit for connection
    time.sleep(0.5)
    
    # 3. Trigger FlexKV transfers
    flexkv_simulation(worker, num_layers)
    
    sglang_thread.join()
    worker.shutdown()

if __name__ == "__main__":
    main()


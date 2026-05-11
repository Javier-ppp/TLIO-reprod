import time
import torch
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from network.model_factory import get_model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def benchmark_model(arch, seq_len, device, warmup_runs=10, benchmark_runs=100):
    net_config = {"in_dim": seq_len // 32 + 1}
    try:
        model = get_model(arch, net_config).to(device)
    except Exception as e:
        print(f"Failed to load {arch} with seq_len {seq_len}: {e}")
        return None

    model.eval()
    
    # Batch size 1, 6 channels, sequence length
    x = torch.randn(1, 6, seq_len).to(device)

    try:
        with torch.no_grad():
            # Warmup
            for _ in range(warmup_runs):
                _ = model(x)
            
            sync(device)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                _ = model(x)
            sync(device)
            end_time = time.perf_counter()
    except Exception as e:
        # Some models (like resnet_seq) crash on specific sequence lengths 
        # due to down/upsampling dimension mismatches.
        return {"arch": arch, "seq_len": seq_len, "error": str(e)}

    total_time = end_time - start_time
    avg_seq_time = total_time / benchmark_runs
    avg_token_time = avg_seq_time / seq_len

    return {
        "arch": arch,
        "seq_len": seq_len,
        "avg_seq_time_ms": avg_seq_time * 1000.0,
        "avg_token_time_ms": avg_token_time * 1000.0
    }

def main():
    device = get_device()
    print(f"Benchmarking on device: {device}\n")

    architectures = ['resnet', 'resnet_seq', 'tcn']
    # 200 is default (1 sec @ 200Hz). 
    # Use powers of 2 for others to avoid resnet_seq dimension issues
    seq_lengths = [200, 512, 1024] 

    print(f"{'Architecture':<15} | {'Seq Len':<10} | {'Seq Time (ms)':<15} | {'Token Time (ms)':<15}")
    print("-" * 65)

    for arch in architectures:
        for seq_len in seq_lengths:
            res = benchmark_model(arch, seq_len, device)
            if res is not None:
                if "error" in res:
                    print(f"{res['arch']:<15} | {res['seq_len']:<10} | {'ERROR':<15} | {res['error']}")
                else:
                    print(f"{res['arch']:<15} | {res['seq_len']:<10} | {res['avg_seq_time_ms']:<15.3f} | {res['avg_token_time_ms']:<15.4f}")
        print("-" * 65)

if __name__ == "__main__":
    main()

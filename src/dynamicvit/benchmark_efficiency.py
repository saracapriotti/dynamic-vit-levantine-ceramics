"""
Efficiency benchmark for the review response: FLOPs, latency, throughput and peak
GPU memory, comparing the trained DynamicViT-LVViT-M/0.7 (token pruning ON) against
its exact static counterpart LV-ViT-M (pruning OFF, same backbone/weights lineage).
"""
import os
import sys
import time

import torch

import config

_DYNVIT_ROOT = os.environ.get("DYNAMICVIT_REPO", config.DYNAMICVIT_REPO)
if _DYNVIT_ROOT not in sys.path:
    sys.path.insert(0, _DYNVIT_ROOT)

from calc_flops import calc_flops  # DynamicViT/calc_flops.py (fvcore-based)

from model import load_model
from model_baselines import build_lvvit


@torch.no_grad()
def measure_latency_throughput(model, device, batch_size=64, img_size=224, n_warmup=20, n_iters=50):
    model.eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for _ in range(n_warmup):
        model(images)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(n_iters):
        model(images)
    torch.cuda.synchronize()
    t1 = time.time()

    throughput = n_iters * batch_size / (t1 - t0)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    # single-image latency
    img1 = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(n_warmup):
        model(img1)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        model(img1)
    torch.cuda.synchronize()
    t1 = time.time()
    latency_ms = 1000.0 * (t1 - t0) / n_iters

    return throughput, peak_mem_mb, latency_ms


def benchmark(name, model, device):
    model.eval()
    model_cpu = model.to("cpu")
    gflops = calc_flops(model_cpu, img_size=224)
    model.to(device)

    throughput, peak_mem_mb, latency_ms = measure_latency_throughput(model, device)

    print(f"\n=== {name} ===")
    print(f"GFLOPs (batch=1):        {gflops:.3f}")
    print(f"Throughput (batch=64):   {throughput:.1f} img/s")
    print(f"Peak GPU memory (b=64):  {peak_mem_mb:.1f} MB")
    print(f"Latency (batch=1):       {latency_ms:.2f} ms")
    return dict(name=name, gflops=gflops, throughput=throughput, peak_mem_mb=peak_mem_mb, latency_ms=latency_ms)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 20

    dvit = load_model(config.BEST_MODEL_PATH, num_classes=num_classes, device=device)
    r_dvit = benchmark("DynamicViT (pruning ON)", dvit, device)
    del dvit
    torch.cuda.empty_cache()

    static_ckpt = os.path.join(config.BASELINES_DIR, "lvvit", "run0", "best_lvvit.pth")
    static = build_lvvit(num_classes=num_classes, device=device)
    static.load_state_dict(torch.load(static_ckpt, map_location=device))
    r_static = benchmark("LV-ViT-M static (pruning OFF)", static, device)
    del static
    torch.cuda.empty_cache()

    print("\n\n=== Summary (reduction of DynamicViT vs static) ===")
    for k in ["gflops", "peak_mem_mb", "latency_ms"]:
        red = 100 * (1 - r_dvit[k] / r_static[k])
        print(f"{k:15s}: {r_dvit[k]:.2f} vs {r_static[k]:.2f}  ->  {red:+.1f}%")
    thr_gain = 100 * (r_dvit["throughput"] / r_static["throughput"] - 1)
    print(f"{'throughput':15s}: {r_dvit['throughput']:.1f} vs {r_static['throughput']:.1f} img/s  ->  {thr_gain:+.1f}%")

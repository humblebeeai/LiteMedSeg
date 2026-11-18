"""Latency / memory benchmark on ONE machine (R1.6, R2.3). Batch 1, 256x256, eval mode, no_grad, fp32.
CPU: all logical threads (PyTorch default). 10 warm-up + 100 timed passes (GPU: 30 + 300) per repeat, 3 repeats, median of per-repeat means.
Module-level breakdown: AblationNet variants (B_Ch=12) with one module removed. Output: results/final_protocol/benchmark_latency.csv, benchmark_modules.csv"""
import csv, platform, statistics as st, sys, time
import torch
sys.path.insert(0, ".")
from scripts.train import build_model
from segmentation_models.models.lite_med_seg_ablation import create_ablation_model
torch.backends.cudnn.benchmark = False
x = torch.randn(1, 3, 256, 256)

def bench(net, dev, warm, n, reps=3):
    net = net.to(dev).eval(); xi = x.to(dev); means = []
    if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(reps):
            for _ in range(warm): net(xi)
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n): net(xi)
            if dev == "cuda": torch.cuda.synchronize()
            means.append((time.perf_counter() - t) / n * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
    return st.median(means), st.pstdev(means), mem

models = ["LiteMedSeg", "UNet", "UNet++", "ResUNet", "ResUNet++", "AttentionUNet", "DeepLabV3+", "SCA-InceptionUNet", "UNeXt", "Fast-SCNN", "MobileViT-Seg"]
rows = []
for m in models:
    net = build_model(m, base_channels=12); p = sum(q.numel() for q in net.parameters()) / 1e6
    c = bench(net, "cpu", 10, 100); g = bench(build_model(m, base_channels=12), "cuda", 30, 300)
    rows.append([m, f"{p:.3f}", f"{c[0]:.2f}", f"{c[1]:.2f}", f"{g[0]:.2f}", f"{g[1]:.2f}", f"{g[2]:.0f}"]); print(rows[-1], flush=True)
with open("results/final_protocol/benchmark_latency.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "params_M", "cpu_ms_median", "cpu_ms_sd_over_repeats", "gpu_ms_median", "gpu_ms_sd_over_repeats", "gpu_peak_mem_MB"]); w.writerows(rows)
V = ["full", "bare", "wo_lsk_none", "wo_eca_none", "wo_mpp_none", "wo_hga_none", "baseline_std"]
mr = []
for v in V:
    net = create_ablation_model(f"Abl-{v}_b12"); p = sum(q.numel() for q in net.parameters()) / 1e6
    c = bench(net, "cpu", 10, 100); g = bench(create_ablation_model(f"Abl-{v}_b12"), "cuda", 30, 300)
    mr.append([v, f"{p:.3f}", f"{c[0]:.2f}", f"{g[0]:.2f}"]); print(mr[-1], flush=True)
with open("results/final_protocol/benchmark_modules.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["variant", "params_M", "cpu_ms", "gpu_ms"]); w.writerows(mr)
print(platform.processor(), torch.__version__, torch.get_num_threads(), torch.cuda.get_device_name(0))

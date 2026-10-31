"""P2 bounded inference engineering profiles.

Compares untrained FULL PREFILL and *reference* cached autoregressive pass
separately. A slow Python token loop is not an optimized decode kernel.
Includes numerical parity, peak CUDA allocated memory, repeat-level raw times.
All lengths/steps bounded for RTX3050 4GB and similarly constrained devices.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from grassmann_edge import TinyLM
from grassmann_edge.paper_reference import PaperGrassmannLM
from grassmann_edge.inference_cache import incremental_tiny, model_state_bytes


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_once(model, ids, kind, mode):
    if mode == "prefill":
        return model(ids)
    if kind == "grassmann":
        return model.forward_incremental(ids)
    return incremental_tiny(model, ids)


def payload_state_bytes(model, kind, context, batch):
    size = next(model.parameters()).element_size()
    if kind == "grassmann":
        return sum(batch * min(context, max(b.offsets)) * b.rank * size
                   for b in model.blocks)
    return model_state_bytes(model, context, batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--lengths", default="16,32,64")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--out", default="results/p2_inference.json")
    a = p.parse_args()
    lengths = [int(x.strip()) for x in a.lengths.split(",")]
    if (not lengths or len(lengths) > 4 or len(set(lengths)) != len(lengths)
            or any(x < 1 or x > 128 for x in lengths)
            or not 2 <= a.repeats <= 15 or not 1 <= a.warmups <= 4):
        p.error("Bounded length/repeat configuration violated")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA not available")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.60)
    output = {
        "status": "UNTRAINED ENGINEERING PROFILE, NOT MODEL PERFORMANCE EVIDENCE",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_hash(),
        "torch": torch.__version__, "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda, "device": a.device,
        "gpu_name": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
        "config": vars(a), "samples": [],
        "notes": [
            "Different parameter counts and architecture operation counts",
            "Full-sequence prefill vs Python-loop cached decode; not equivalent to production kernels",
            "Random model weights; no language modeling quality measured",
            "CUDA peak is torch.max_memory_allocated, not total VRAM process usage",
            "Energy/watts per token not measured; GPU power limit unmodified",
            "Causal state payload byte estimates exclude Python/tensor/container overhead",
        ],
    }
    with torch.inference_mode():
        for length in lengths:
            for kind in ("grassmann", "transformer", "local"):
                torch.manual_seed(99)
                if kind == "grassmann":
                    model = PaperGrassmannLM(vocab=32, dim=32, layers=2,
                                             rank=8, context=length,
                                             offsets=(1, 2, 4, 8))
                else:
                    model = TinyLM(kind=kind, vocab=32, dim=32, layers=2,
                                   rank=8, context=length,
                                   offsets=(1, 2, 4, 8))
                model = model.to(a.device).eval()
                ids = torch.randint(0, 32, (1, length), device=a.device)
                full = run_once(model, ids, kind, "prefill")
                cached = run_once(model, ids, kind, "cached_decode")
                torch.testing.assert_close(full, cached, atol=3e-5, rtol=3e-5)
                maximum_error = (full - cached).abs().max().item()
                for mode in ("prefill", "cached_decode"):
                    for _ in range(a.warmups):
                        run_once(model, ids, kind, mode)
                    synchronize(a.device)
                    if a.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(a.repeats):
                        synchronize(a.device)
                        t = time.perf_counter()
                        run_once(model, ids, kind, mode)
                        synchronize(a.device)
                        times.append((time.perf_counter() - t) * 1000)
                    record = {
                        "kind": kind, "mode": mode, "length": length,
                        "params": sum(p.numel() for p in model.parameters()),
                        "median_ms": statistics.median(times),
                        "min_ms": min(times), "max_ms": max(times),
                        "raw_ms": times, "cached_parity_abs_error": maximum_error,
                        "theoretical_cache_payload_bytes": payload_state_bytes(
                            model, kind, length, 1) if mode == "cached_decode" else None,
                        "torch_peak_cuda_allocated_bytes": (
                            torch.cuda.max_memory_allocated()
                            if a.device == "cuda" else None),
                    }
                    output["samples"].append(record)
                    print(f"{kind} {mode} L={length} "
                          f"median_ms={record['median_ms']:.4f}",
                          flush=True)
                del model, ids, full, cached
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()

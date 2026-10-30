"""Untrained full-forward smoke timing; NOT inference quality or decode comparison."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from grassmann_edge import TinyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--lengths", default="16,32,64")
    p.add_argument("--out", default="results/untrained_forward.json")
    args = p.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if not lengths or any(x < 1 or x > 128 for x in lengths):
        p.error("Lengths restricted to 1..128")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            p.error("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.6)
    torch.set_num_threads(4)
    runs = []
    for length in lengths:
        for kind in ("grassmann", "transformer", "local"):
            torch.manual_seed(7)
            m = TinyLM(kind, dim=32, rank=8, layers=2, context=length).to(args.device).eval()
            x = torch.randint(0, 32, (1, length), device=args.device)
            with torch.inference_mode():
                for _ in range(2):
                    m(x)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                timings = []
                for _ in range(5):
                    start = time.perf_counter()
                    m(x)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    timings.append((time.perf_counter() - start) * 1000)
            item = {"kind": kind, "length": length, "median_ms": statistics.median(timings),
                    "samples_ms": timings}
            runs.append(item)
            print(kind, "context", length, "median_ms", round(item["median_ms"], 3))
            del m, x
    result = {"warning": "randomly initialized full forward ONLY; not cached decode, not fair speed comparison",
              "device": args.device, "torch": torch.__version__, "runs": runs}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

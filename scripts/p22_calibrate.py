"""P2.2 bounded CUDA calibration with parameter-matched reference controls.

All candidates are small trial runs only. Do not call them trained LMs or
performance conclusions. Each kind uses identical d, layers, context, rank,
offsets, batch and data. Feedforward width is explicitly varied to match
learnable parameter counts to a Transformer control within 2% tolerance.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from grassmann_edge import TinyLM
from grassmann_edge.paper_reference import PaperGrassmannLM
from scripts.controlled_realtext import load_split, batch_from

KINDS = ("grassmann", "transformer", "local")


def create_model(kind, dim, layers, rank, context, offsets, vocab=256, ff_width=None):
    kw = dict(vocab=vocab, dim=dim, layers=layers, rank=rank,
              context=context, offsets=offsets)
    model = PaperGrassmannLM(**kw) if kind == "grassmann" else TinyLM(kind=kind, **kw)
    if ff_width is not None:
        for block in model.blocks:
            block.ff = nn.Sequential(nn.Linear(dim, ff_width),
                                     nn.GELU(), nn.Linear(ff_width, dim))
    return model


def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_widths(dim, layers, rank, context, offsets):
    target = params(create_model("transformer", dim, layers, rank, context, offsets))
    result = {"transformer": 4 * dim}
    for kind in ("grassmann", "local"):
        base = params(create_model(kind, dim, layers, rank, context, offsets, ff_width=4 * dim))
        # Every +1 FF width adds approximately layers*(2*dim+1) params.
        step = layers * (2 * dim + 1)
        guess = max(16, (4 * dim) + round((target - base) / step))
        candidates = range(max(16, guess-8), guess+9)
        best = min(candidates, key=lambda w: abs(
            params(create_model(kind, dim, layers, rank, context, offsets, ff_width=w)) - target))
        result[kind] = best
    counts = {k: params(create_model(k, dim, layers, rank, context, offsets,
                                      ff_width=result[k])) for k in KINDS}
    mismatch = (max(counts.values()) - min(counts.values())) / max(counts.values())
    if mismatch > 0.02:
        raise ValueError(f"Model matching exceeds 2% tolerance: {counts}")
    return result, counts, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (64 <= a.dim <= 512 and a.dim % 4 == 0 and
            2 <= a.layers <= 8 and 8 <= a.rank <= 48 and
            1 <= a.batch <= 12 and 64 <= a.context <= 768 and
            1 <= a.steps <= 8 and 1 <= len(a.kinds) <= 3):
        ap.error("Outside predeclared safe calibration menu")
    if not torch.cuda.is_available():
        ap.error("CUDA unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.cuda.set_per_process_memory_fraction(0.60)
    offsets = (1, 2, 4, 8, 16, 32)
    widths, counts, mismatch = matched_widths(
        a.dim, a.layers, a.rank, a.context, offsets)
    provenance_path = Path("data/wikitext-2-raw-v1/PROVENANCE.json")
    provenance = json.loads(provenance_path.read_text())
    train = load_split(provenance_path.parent, "train", provenance)
    result = {"status": "CALIBRATION ONLY", "config": vars(a),
              "ff_widths": widths, "parameter_counts": counts,
              "parameter_mismatch_fraction": mismatch,
              "precision": "fp32", "offsets": list(offsets),
              "runs": [], "failure": None}
    try:
        for kind in a.kinds:
            torch.cuda.empty_cache()
            torch.manual_seed(2026)
            model = create_model(kind, a.dim, a.layers, a.rank,
                                 a.context, offsets, ff_width=widths[kind]).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            gen = torch.Generator().manual_seed(2026)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for step in range(a.steps):
                x = batch_from(train, a.context, a.batch, gen, "cuda")
                pred = model(x[:, :-1])
                loss = F.cross_entropy(pred.reshape(-1, 256), x[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    raise ArithmeticError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_alloc = torch.cuda.max_memory_allocated()/2**20
            peak_reserved = torch.cuda.max_memory_reserved()/2**20
            result["runs"].append({
                "kind": kind, "loss": float(loss.item()),
                "elapsed_seconds": elapsed, "step_seconds": elapsed/a.steps,
                "peak_allocated_mib": round(peak_alloc, 2),
                "peak_reserved_mib": round(peak_reserved, 2),
                "parameters": counts[kind],
            })
            print("CAL", kind, "params", counts[kind], "step_s", round(elapsed/a.steps,3),
                  "peak_alloc_mib", round(peak_alloc,1), "peak_reserved_mib", round(peak_reserved,1),
                  flush=True)
            del model, optimizer, x, pred, loss
            torch.cuda.empty_cache()
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {str(exc)[:350]}"
        print("CALIBRATION_FAILED", result["failure"], flush=True)
    finally:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print("RESULT", out, flush=True)
    if result["failure"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

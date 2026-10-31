"""Bounded real-text byte-level pilot; NOT WordPiece or paper benchmark.

Uses only SHA-256 checked train/validation WikiText-2 data. Never loads test.
Three models share the same predetermined training sample positions, seeds,
optimization budget, optimizer, and validation sample positions.
Parameter sizes and compute workloads are NOT precisely matched.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from grassmann_edge import TinyLM
from grassmann_edge.paper_reference import PaperGrassmannLM


KINDS = ("grassmann", "transformer", "local")


def load_split(folder, name, provenance):
    raw = (folder / f"{name}.bytes").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != provenance["splits"][name]["prepared_sha256"]:
        raise ValueError(f"Prepared {name} hash mismatch")
    # memory efficient: avoid constructing an 11-million-element Python list
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dtype=torch.long)


def batch_from(stream, length, batch, generator, device):
    starts = torch.randint(0, len(stream) - length - 1, (batch,), generator=generator)
    slices = [stream[int(i):int(i)+length+1] for i in starts]
    return torch.stack(slices, dim=0).to(device)


def new_model(kind, length):
    kwargs = {"vocab": 256, "dim": 32, "layers": 2, "rank": 8,
              "context": length, "offsets": (1, 2, 4, 8)}
    return PaperGrassmannLM(**kwargs) if kind == "grassmann" else TinyLM(kind=kind, **kwargs)


def loss_on(model, tokens):
    prediction = model(tokens[:, :-1])
    return F.cross_entropy(prediction.reshape(-1, 256), tokens[:, 1:].reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--out", default="results/lenovo_p1_realtext.json")
    a = parser.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if not (1 <= len(seeds) <= 5 and len(set(seeds)) == len(seeds)
            and all(0 <= i <= 100000 for i in seeds)
            and 1 <= a.steps <= 150 and 1 <= a.batch <= 8
            and 16 <= a.length <= 128 and 1 <= a.eval_batches <= 8):
        parser.error("Invalid or unsafe configuration")
    if a.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.60)
    folder = Path("data/wikitext-2-raw-v1")
    provenance = json.loads((folder/"PROVENANCE.json").read_text(encoding="utf-8"))
    train = load_split(folder, "train", provenance)
    validation = load_split(folder, "validation", provenance)
    if min(train.numel(), validation.numel()) < a.length + 2:
        parser.error("Corpus is too short")
    params = {k: sum(p.numel() for p in new_model(k, a.length).parameters()) for k in KINDS}
    result = {
        "status": "EXPLORATORY REAL-TEXT PILOT, NO PAPER/WORDPIECE COMPARISON",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "device": a.device,
        "config": vars(a), "tokenization": "UTF-8 bytes, vocab=256",
        "upstream_revision": provenance["revision"],
        "train_sha256": provenance["splits"]["train"]["prepared_sha256"],
        "validation_sha256": provenance["splits"]["validation"]["prepared_sha256"],
        "test_split_used": False,
        "parameter_counts": params,
        "parameter_mismatch_fraction": (max(params.values())-min(params.values()))/max(params.values()),
        "training_protocol": "equal seeds, samples, steps, AdamW lr=0.002, wd=0.01; not FLOP matched",
        "runs": [], "summary": {},
    }
    for seed in seeds:
        for kind in KINDS:
            torch.manual_seed(seed)
            model = new_model(kind, a.length).to(a.device)
            optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
            generator = torch.Generator().manual_seed(123456 + seed)
            losses = []
            model.train()
            started = time.perf_counter()
            for step in range(a.steps):
                tokens = batch_from(train, a.length, a.batch, generator, a.device)
                loss = loss_on(model, tokens)
                if not torch.isfinite(loss):
                    raise ArithmeticError(f"Non-finite {kind} seed={seed} step={step}")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                losses.append(float(loss.item()))
            if a.device == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter()-started
            model.eval()
            valid_generator = torch.Generator().manual_seed(987654 + seed)
            val_losses = []
            with torch.inference_mode():
                for i in range(a.eval_batches):
                    tokens = batch_from(validation, a.length, a.batch, valid_generator, a.device)
                    val_losses.append(float(loss_on(model, tokens).item()))
            val_nats = statistics.mean(val_losses)
            entry = {
                "kind":kind,"seed":seed,"parameters":params[kind],
                "train_tokens":a.steps*a.batch*a.length,
                "train_losses":losses,"train_seconds":duration,
                "validation_loss_nats":val_nats,
                "validation_bits_per_byte":val_nats / math.log(2)
            }
            result["runs"].append(entry)
            print(f"{kind} seed={seed} bits_per_byte={entry['validation_bits_per_byte']:.5f}",flush=True)
            del model, optim
    for kind in KINDS:
        values = [x["validation_bits_per_byte"] for x in result["runs"] if x["kind"]==kind]
        result["summary"][kind]={"n":len(values),"mean_bits_per_byte":statistics.mean(values),
                                "sample_sd":statistics.stdev(values) if len(values)>1 else None}
    out=Path(a.out)
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("SAVED",out,flush=True)


if __name__=="__main__":
    main()

"""P2.2 deterministic, checkpointed, parameter-matched small-LM training block.

Run one seed/model/finite block at a time, only through p22_guard.
Checkpoints are private to ignored results/, raw numeric research metrics
can be released after review. This is an exploratory byte-LM experiment,
not original-paper WordPiece perplexity reproduction.
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

from scripts.controlled_realtext import load_split, batch_from
from scripts.p22_calibrate import create_model, matched_widths


def atomic_torch_save(value, path):
    tmp = path.with_suffix(".partial")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("grassmann","transformer","local"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stop-step", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--dim", type=int, default=288)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--rank", type=int, default=28)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--out-dir", default="results/p22/train")
    a = p.parse_args()
    if not (a.seed in (11,17,23,31,47)
            and 1 <= a.stop_step <= 2000
            and 5 <= a.checkpoint_every <= 100
            and (a.dim,a.layers,a.rank,a.batch,a.context) ==
                (288,6,28,8,512) and 0 < a.lr <= 0.001):
        p.error("Not part of predeclared safe P2.2 protocol")
    if not torch.cuda.is_available():
        p.error("CUDA missing")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.cuda.set_per_process_memory_fraction(0.60)
    offsets = (1,2,4,8,16,32)
    widths, counts, mismatch = matched_widths(a.dim,a.layers,a.rank,a.context,offsets)
    if mismatch > 0.02:
        p.error("Parameter mismatch >2%")
    provenance_path = Path("data/wikitext-2-raw-v1/PROVENANCE.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    train = load_split(provenance_path.parent, "train", provenance)
    validation = load_split(provenance_path.parent, "validation", provenance)
    destination = Path(a.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f"{a.kind}_seed{a.seed}_d{a.dim}_ctx{a.context}"
    checkpoint_path = destination / (prefix + ".ckpt.pt")
    metrics_path = destination / (prefix + ".metrics.json")
    torch.manual_seed(a.seed)
    model = create_model(a.kind,a.dim,a.layers,a.rank,a.context,offsets,
                         ff_width=widths[a.kind]).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    generator = torch.Generator().manual_seed(123456+a.seed)
    step_start = 0
    losses = []
    if checkpoint_path.exists():
        # Only trust checkpoints we generated locally, never public artifacts.
        ckpt = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        if ckpt["config"] != vars(a) | {"stop_step": ckpt["config"]["stop_step"]}:
            raise ValueError("Checkpoint config mismatch")
        if ckpt["dataset_revision"] != provenance["revision"] or \
           ckpt["train_sha256"] != provenance["splits"]["train"]["prepared_sha256"]:
            raise ValueError("Checkpoint dataset identity mismatch")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        generator.set_state(ckpt["train_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
        step_start=ckpt["step"]
        losses=ckpt["losses"]
        print("RESUMED",prefix,step_start,flush=True)
    if step_start > a.stop_step:
        raise ValueError("Refusing backward checkpoint step")
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    model.train()
    for step in range(step_start,a.stop_step):
        x = batch_from(train,a.context,a.batch,generator,"cuda")
        prediction = model(x[:,:-1])
        loss = F.cross_entropy(prediction.reshape(-1,256),x[:,1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ArithmeticError(f"Nonfinite loss {a.kind} {a.seed} {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if (step+1)%10==0:
            torch.cuda.synchronize()
            print("STEP",prefix,step+1,"loss",round(losses[-1],5),
                  "peakMiB",round(torch.cuda.max_memory_reserved()/2**20,1),flush=True)
        if (step+1)%a.checkpoint_every==0 or step+1==a.stop_step:
            ckpt={
                "config": vars(a), "step":step+1,"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"train_rng":generator.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state(),"losses":losses,
                "dataset_revision":provenance["revision"],
                "train_sha256":provenance["splits"]["train"]["prepared_sha256"],
            }
            atomic_torch_save(ckpt,checkpoint_path)
            print("CHECKPOINT",step+1,flush=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-wall
    model.eval()
    val_generator=torch.Generator().manual_seed(987654+a.seed)
    vals=[]
    with torch.inference_mode():
        for _ in range(8):
            x=batch_from(validation,a.context,a.batch,val_generator,"cuda")
            pred=model(x[:,:-1])
            vals.append(float(F.cross_entropy(pred.reshape(-1,256),x[:,1:].reshape(-1)).item()))
    out={
        "status":"EXPLORATORY FP32 BYTE-LM P2.2; NOT PAPER REPRODUCTION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind":a.kind,"seed":a.seed,"config":vars(a),
        "offsets":list(offsets),"ff_widths":widths,"parameter_counts":counts,
        "parameter_mismatch_fraction":mismatch,
        "train_tokens_total":a.stop_step*a.context*a.batch,
        "steps_completed":a.stop_step,"steps_resumed_from":step_start,
        "train_losses_nats":losses,
        "mean_validation_bits_per_byte":statistics.mean(vals)/math.log(2),
        "validation_per_batch_nats":vals,"block_wall_seconds":elapsed,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,
        "dataset_revision":provenance["revision"],
        "train_sha256":provenance["splits"]["train"]["prepared_sha256"],
        "validation_sha256":provenance["splits"]["validation"]["prepared_sha256"],
        "test_split_used":False,
        "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    }
    metrics_path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("COMPLETE",prefix,"step",a.stop_step,"validation_bpb",
          round(out["mean_validation_bits_per_byte"],6),
          "peak_alloc_mib",round(out["peak_allocated_mib"],1),
          "checkpoint_hash",out["checkpoint_sha256"][:12],flush=True)


if __name__=="__main__":
    main()

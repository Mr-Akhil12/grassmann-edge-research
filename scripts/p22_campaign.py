"""P2.2 finite, restartable, locally guarded training campaign.

Not a background scheduler. Runs a finite number of guarded subprocesses,
with explicit overall wall-clock cap and STOP_REQUESTED file interrupt.
All checkpoint and telemetry files are in ignored results/p22.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "p22"
TRAIN = BASE / "train"
STOP = BASE / "STOP_REQUESTED"
LOCK = BASE / "campaign.lock"
KINDS = ("grassmann", "transformer", "local")


def progress(kind, seed):
    src = TRAIN / f"{kind}_seed{seed}_d288_ctx512.metrics.json"
    if not src.is_file():
        return 0
    record = json.loads(src.read_text(encoding="utf-8"))
    if record["kind"] != kind or record["seed"] != seed:
        raise ValueError(f"Corrupt metrics {src}")
    return int(record["steps_completed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-steps", type=int, default=100)
    ap.add_argument("--seeds", default="11,17,23")
    ap.add_argument("--block-steps", type=int, default=100)
    ap.add_argument("--max-minutes", type=int, default=20)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if not (a.target_steps >= 50 and a.target_steps <= 1000
            and a.block_steps >= 20 and a.block_steps <= 100
            and 5 <= a.max_minutes <= 120
            and len(seeds) <= 5 and len(set(seeds)) == len(seeds)
            and all(x in (11,17,23,31,47) for x in seeds)):
        ap.error("Outside bounded P2.2 campaign protocol")
    BASE.mkdir(parents=True, exist_ok=True)
    if STOP.exists():
        raise SystemExit("STOP_REQUESTED marker present; no work started")
    started = time.monotonic()
    completed = []
    initial_progress = {f"{s}-{k}": progress(k,s) for s in seeds for k in KINDS}
    try:
        with LOCK.open("x", encoding="utf-8") as f:
            json.dump({"start_utc":datetime.now(timezone.utc).isoformat(),
                       "pid":os.getpid(),"target_steps":a.target_steps}, f)
    except FileExistsError:
        raise SystemExit("Campaign lock exists. Verify no existing run before manual cleanup")
    try:
        # Equalize temporal order by rotating model order by seed / block.
        for round_no in range((a.target_steps + a.block_steps - 1)//a.block_steps):
            for seed_pos, seed in enumerate(seeds):
                order = KINDS[(round_no+seed_pos)%3:] + KINDS[:(round_no+seed_pos)%3]
                for kind in order:
                    now = progress(kind, seed)
                    if now >= a.target_steps:
                        continue
                    if STOP.exists():
                        raise RuntimeError("STOP_REQUESTED from owner")
                    if time.monotonic()-started > a.max_minutes*60:
                        raise RuntimeError("Campaign overall wall-clock limit reached")
                    stop_step = min(a.target_steps, max(
                        (round_no+1)*a.block_steps, now))
                    if stop_step <= now:
                        continue
                    cmd=[sys.executable,"-m","scripts.p22_guard","--mode","train",
                         "--dim","288","--layers","6","--rank","28",
                         "--batch","8","--context","512",
                         "--kind",kind,"--seed",str(seed),
                         "--stop-step",str(stop_step),"--seconds","180"]
                    print("P22 CAMPAIGN",kind,seed,now,"->",stop_step,flush=True)
                    rc=subprocess.call(cmd,cwd=ROOT)
                    completed.append({"kind":kind,"seed":seed,"from":now,
                                      "to":stop_step,"exit_code":rc})
                    if rc:
                        raise RuntimeError(f"Guarded trial failed: {kind} seed={seed}")
        print("P22 CAMPAIGN COMPLETE",len(completed),"blocks",flush=True)
    finally:
        report={"at_utc":datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds":time.monotonic()-started,
                "initial_progress":initial_progress,
                "target_steps":a.target_steps,"completed_blocks":completed,
                "final_progress":{f"{s}-{k}":progress(k,s) for s in seeds for k in KINDS}}
        (BASE / f"campaign_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
            json.dumps(report,indent=2)+"\n",encoding="utf-8")
        LOCK.unlink(missing_ok=True)


if __name__=="__main__":
    main()

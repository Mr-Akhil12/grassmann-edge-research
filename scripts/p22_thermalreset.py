"""P2.2c: finite fresh-seed training with mandatory 15-minute thermal resets.

Usage:
  python -m scripts.p22_thermalreset                 # inspect planned work, NO training
  python -m scripts.p22_thermalreset --execute       # explicitly authorized 3-run protocol

Safety and scientific invariants:
 - Preserve original results/p22/STOP_REQUESTED; this is a NEW, scoped
   one-off finite controller, never a general bypass of the stop.
 - Three fresh architecture runs (seed 11, each 100 AdamW steps), never resume
   or overwrite prior P2.2 checkpoint history.
 - Minimum 900-second rest AFTER EACH training block and BEFORE next one,
   plus GPU <=38C, idle <=10% and <=128 MiB VRAM for 60 consecutive seconds.
 - Initial trial also requires 60 sec verified cold, low-utilization baseline.
 - Raise if cooldown exceeds 25 minutes; never waive any check.
 - Abort child at >=78C, >=3000 MiB GPU memory, <8 GiB free RAM,
   loss of AC, loss of telemetry, 150-second child limit, or owner ABORT_NOW.
 - One child at a time, 55-minute total-session ceiling. No automated retry.
 - Raw telemetry, stdout, stderr, progress, and failed runs all saved privately
   to ignored results/p22/thermalreset_v1; publish after review.

A 15 minute rest is a cooling protocol, not proof of CPU/heatsink equilibrium.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from scripts.p22_guard import sensors, ensure_safe

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "p22"
OUT = BASE / "thermalreset_v1"
TRAIN_DIR = OUT / "train"
RUNS = ("grassmann", "transformer", "local")
MIN_REST_SECONDS = 15 * 60
STABLE_SECONDS = 60
COOL_TIMEOUT_SECONDS = 25 * 60
MAX_CHILD_SECONDS = 150
MAX_SESSION_SECONDS = 55 * 60
COLD_TEMP_C = 38
IDLE_UTIL_MAX = 10
IDLE_VRAM_MAX_MIB = 128
POLL_SECONDS = 2


def now():
    return datetime.now(timezone.utc).isoformat()


def abort_if_requested():
    if (BASE / "ABORT_NOW").exists() or (OUT / "ABORT_NOW").exists():
        raise RuntimeError("User ABORT_NOW marker present")


def save_json_atomic(dest, contents):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".partial")
    temp.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, dest)


def preflight():
    if not (BASE / "STOP_REQUESTED").is_file():
        raise RuntimeError("Expected original STOP_REQUESTED marker to stay in place")
    if (BASE / "campaign.lock").exists():
        raise RuntimeError("Previous P2.2 campaign lock exists")
    abort_if_requested()
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Thermal reset output directory is not empty; refusing overwrite")
    for kind in RUNS:
        stem = f"{kind}_seed11_d288_ctx512"
        if (TRAIN_DIR / f"{stem}.ckpt.pt").exists() or (TRAIN_DIR / f"{stem}.metrics.json").exists():
            raise RuntimeError(f"Previous training material exists for {kind}")
    # Fails closed for power, NVIDIA telemetry, and physical RAM.
    return ensure_safe(sensors())


def take_sample(phase, started, trace):
    abort_if_requested()
    if time.monotonic() - started >= MAX_SESSION_SECONDS:
        raise TimeoutError("Total session exceeded 55-minute hard limit")
    t = ensure_safe(sensors())
    t["phase"] = phase
    t["session_elapsed_s"] = round(time.monotonic() - started, 2)
    trace.append(t)
    return t


def idle_enough(t):
    return (t["gpu_temp_c"] <= COLD_TEMP_C
            and t["gpu_util_percent"] <= IDLE_UTIL_MAX
            and t["gpu_vram_mib"] <= IDLE_VRAM_MAX_MIB)


def thermal_reset(phase, session_started, prior_finished, trace, manifest_path, manifest):
    start = time.monotonic()
    stable_since = None
    last_report = start
    last_write = start
    while True:
        t = take_sample(phase, session_started, trace)
        current = time.monotonic()
        rest_elapsed = (current - prior_finished) if prior_finished is not None else None
        if idle_enough(t):
            if stable_since is None:
                stable_since = current
        else:
            stable_since = None
        stable_for = (current - stable_since) if stable_since is not None else 0
        # Must achieve BOTH mandatory rest and continuous cold stability.
        rested = prior_finished is None or rest_elapsed >= MIN_REST_SECONDS
        if rested and stable_for >= STABLE_SECONDS:
            print("THERMAL_RESET_OK", phase, "C", t["gpu_temp_c"],
                  "rest_s", round(rest_elapsed or 0, 1),
                  "cold_stable_s", round(stable_for, 1), flush=True)
            manifest["gate_history"].append({
                "phase": phase, "at_utc": now(), "gpu_c": t["gpu_temp_c"],
                "rest_elapsed_s": rest_elapsed, "cold_stable_s": stable_for,
                "cold_util_pct": t["gpu_util_percent"],
                "idle_vram_mib": t["gpu_vram_mib"]
            })
            save_json_atomic(manifest_path, manifest)
            return
        if current - start > COOL_TIMEOUT_SECONDS:
            raise TimeoutError(f"Thermal reset {phase} did not complete in 25 minutes")
        if current - last_report >= 30:
            print("COOLING", phase, "gpuC", t["gpu_temp_c"],
                  "VRAM", t["gpu_vram_mib"], "util", t["gpu_util_percent"],
                  "mandatory_rest_remaining_s",
                  max(0, round(MIN_REST_SECONDS - (rest_elapsed or 0))) if prior_finished is not None else 0,
                  "cold_stable_s", round(stable_for), flush=True)
            last_report = current
        if current - last_write >= 10:
            manifest["latest_gate"] = {
                "phase": phase, "gpu_c": t["gpu_temp_c"],
                "rest_elapsed_s": rest_elapsed, "cold_stable_s": stable_for}
            save_json_atomic(manifest_path, manifest)
            save_json_atomic(OUT/"THERMAL_SAMPLES.json", {"samples": trace})
            last_write = current
        time.sleep(POLL_SECONDS)


def train_once(kind, session_started, trace, manifest, manifest_path):
    if kind not in RUNS:
        raise ValueError("Undeclared model")
    t = take_sample(f"start_{kind}", session_started, trace)
    if not idle_enough(t):
        raise RuntimeError("GPU no longer at cold baseline when launching model")
    stem = f"{kind}_seed11_d288_ctx512"
    if (TRAIN_DIR / f"{stem}.ckpt.pt").exists():
        raise RuntimeError("Fresh-run checkpoint already exists")
    command = [
        sys.executable, "-m", "scripts.p22_train", "--kind", kind,
        "--seed", "11", "--stop-step", "100", "--checkpoint-every", "20",
        "--out-dir", str(TRAIN_DIR.relative_to(ROOT)).replace("\\", "/")
    ]
    stdout_path = OUT / f"{kind}_seed11.stdout.txt"
    stderr_path = OUT / f"{kind}_seed11.stderr.txt"
    record = {
        "kind": kind, "seed": 11, "start_utc": now(), "start_temp_c": t["gpu_temp_c"],
        "command": command[1:], "finished_utc": None, "duration_s": None,
        "exit_code": None, "failure": None,
        "stdout": stdout_path.name, "stderr": stderr_path.name
    }
    manifest["runs"].append(record)
    save_json_atomic(manifest_path, manifest)
    start = time.monotonic()
    child = None
    try:
        with stdout_path.open("w", encoding="utf-8") as o, stderr_path.open("w", encoding="utf-8") as e:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                      stdout=o, stderr=e,
                      env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            print("RUNNING", kind, "seed11", "C", t["gpu_temp_c"], flush=True)
            last_report = start
            last_write = start
            while child.poll() is None:
                time.sleep(POLL_SECONDS)
                latest = take_sample(f"training_{kind}", session_started, trace)
                elapsed = time.monotonic() - start
                if elapsed > MAX_CHILD_SECONDS:
                    raise TimeoutError(f"{kind}: hard child runtime exceeded 150sec")
                if elapsed - (last_report - start) >= 10:
                    print("ACTIVE", kind, "elapsed", round(elapsed, 1),
                          "GPU_C", latest["gpu_temp_c"],
                          "gpuW", latest["nvidia_power_w"],
                          "VRAM", latest["gpu_vram_mib"], flush=True)
                    last_report = time.monotonic()
                if time.monotonic() - last_write >= 10:
                    manifest["current_run_sample"] = latest
                    save_json_atomic(manifest_path, manifest)
                    save_json_atomic(OUT/"THERMAL_SAMPLES.json", {"samples": trace})
                    last_write = time.monotonic()
            record["exit_code"] = child.returncode
            if child.returncode != 0:
                raise RuntimeError(f"{kind}: Python training returned {child.returncode}")
    except BaseException as ex:
        record["failure"] = f"{type(ex).__name__}: {ex}"
        print("GUARD_STOP",kind,record["failure"],flush=True)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
            record["exit_code"] = child.returncode
        record["duration_s"] = round(time.monotonic() - start, 3)
        record["finished_utc"] = now()
        save_json_atomic(manifest_path,manifest)
        print("FINISHED",kind,"seconds",record["duration_s"],
              "exit",record["exit_code"],"failure",record["failure"],flush=True)
    metrics = TRAIN_DIR / f"{stem}.metrics.json"
    if not metrics.exists():
        raise RuntimeError(f"{kind}: completed with no metrics")
    summary = json.loads(metrics.read_text(encoding="utf-8"))
    if summary.get("steps_completed") != 100 or summary.get("steps_resumed_from") != 0:
        raise RuntimeError(f"{kind}: unexpected checkpoint history")
    record["bits_per_byte"] = summary["mean_validation_bits_per_byte"]
    record["parameters"] = summary["parameter_counts"][kind]
    record["peak_allocated_mib"] = summary["peak_allocated_mib"]
    record["peak_reserved_mib"] = summary["peak_reserved_mib"]
    save_json_atomic(manifest_path,manifest)
    return time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    print("PROTOCOL",{
        "models": RUNS, "seed": 11, "steps_each": 100,
        "minimum_between_runs_minutes": 15,
        "cold_gpu_c_max": COLD_TEMP_C, "cold_stable_seconds": STABLE_SECONDS,
        "gpu_abort_c":78, "max_child_seconds":MAX_CHILD_SECONDS,
        "max_session_minutes":MAX_SESSION_SECONDS//60,
        "historical_STOP_REQUESTED_retained": True},flush=True)
    if not a.execute:
        print("DRY RUN ONLY. Use --execute to conduct exactly this finite comparison.",flush=True)
        return
    state = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    exclusive = OUT / "run.lock"
    with exclusive.open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid":os.getpid(),"at_utc":now()}))
    session_started = time.monotonic()
    manifest_path = OUT / "MANIFEST.json"
    manifest = {
        "status": "running", "started_utc": now(),
        "source_git_revision": "da07db981dd1c7b1f003c1904f95474a555dcd4c",
        "protocol": "one-time 3 model seed-11 fresh 100-step thermal-reset comparison",
        "first_hardware": state,
        "mandatory_rest_seconds": MIN_REST_SECONDS, "cold_stable_seconds": STABLE_SECONDS,
        "max_gpu_temp_c": 78, "max_gpu_vram_mib":3000,
        "minimum_free_ram_gib":8, "ac_required":True,
        "sampling_seconds":POLL_SECONDS, "runs":[], "gate_history":[],
        "latest_gate":None,"current_run_sample":None,"failure":None
    }
    trace = []
    prior_finished = None
    try:
        save_json_atomic(manifest_path, manifest)
        for kind in RUNS:
            thermal_reset(f"before_{kind}", session_started, prior_finished,
                          trace, manifest_path, manifest)
            prior_finished = train_once(kind,session_started,trace,manifest,manifest_path)
            print("NEXT: mandatory 15min rest",flush=True)
        manifest["status"] = "completed"
        print("ALL_THREE_COMPLETED",flush=True)
    except BaseException as e:
        manifest["status"]="aborted"
        manifest["failure"]=f"{type(e).__name__}: {e}"
        print("SESSION_ABORT",manifest["failure"],flush=True)
        raise
    finally:
        manifest["finished_utc"]=now()
        manifest["session_seconds"]=round(time.monotonic()-session_started,2)
        save_json_atomic(manifest_path,manifest)
        save_json_atomic(OUT/"THERMAL_SAMPLES.json", {"samples": trace})
        # DO NOT delete the historical STOP_REQUESTED marker.
        exclusive.unlink(missing_ok=True)
        print("SAVED",manifest_path,"status",manifest["status"],flush=True)


if __name__ == "__main__":
    main()

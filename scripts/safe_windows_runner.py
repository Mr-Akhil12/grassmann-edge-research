"""Windows-safe, bounded experiment supervisor; no PowerShell policies changed.

Invocations:
  python -m scripts.safe_windows_runner --mode inspect
  python -m scripts.safe_windows_runner --mode p1
  python -m scripts.safe_windows_runner --mode p2
Supervises exactly one child process at a time. Kills only children it starts.
A loss of GPU telemetry, memory headroom, or time budget triggers termination.
"""
import argparse
import ctypes
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_free_gib():
    if os.name != "nt":
        raise RuntimeError("Windows-only supervisor")
    ms = MemoryStatus()
    ms.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        raise RuntimeError("System memory telemetry unavailable")
    return ms.ullAvailPhys / (1024**3)


def gpu_values():
    command = [
        "nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    r = subprocess.run(command, capture_output=True, text=True, timeout=7, check=True)
    line = r.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise RuntimeError("Unexpected NVIDIA telemetry")
    return tuple(float(x) for x in parts)


def guard():
    free = memory_free_gib()
    temp, vram, utilization = gpu_values()
    if temp >= 80:
        raise RuntimeError(f"GPU temperature >=80 C ({temp:.1f})")
    if vram >= 3400:
        raise RuntimeError(f"GPU memory >=3400 MiB ({vram:.0f})")
    if free < 8.0:
        raise RuntimeError(f"Free RAM under 8 GiB ({free:.2f})")
    return {"gpu_c": temp, "gpu_mib": vram, "gpu_pct": utilization,
            "free_ram_gib": round(free, 2)}


def runs_for_mode(mode):
    if mode == "inspect":
        return [
            [sys.executable, "-m", "pytest", "-q"],
        ]
    if mode == "p1":
        return [
            [sys.executable, "-m", "scripts.controlled_synthetic",
             "--device", "cuda", "--steps", "60", "--seeds", "11,17,23",
             "--lag", "4", "--out", "results/lenovo_p1_lag4.json",
             "--max-seconds", "220"],
            [sys.executable, "-m", "scripts.controlled_synthetic",
             "--device", "cuda", "--steps", "60", "--seeds", "11,17,23",
             "--lag", "24", "--out", "results/lenovo_p1_lag24.json",
             "--max-seconds", "220"],
        ]
    if mode == "p1text":
        return [
            [sys.executable, "-m", "scripts.controlled_realtext", "--device", "cuda",
             "--steps", "60", "--seeds", "11,17,23",
             "--out", "results/lenovo_p1_realtext.json"],
        ]
    if mode == "p2":
        return [
            [sys.executable, "-m", "scripts.benchmark_p2",
             "--device", "cuda", "--lengths", "16,32,64,128",
             "--warmups", "2", "--repeats", "5",
             "--out", "results/lenovo_p2_gpu.json"],
            [sys.executable, "-m", "scripts.benchmark_p2",
             "--device", "cpu", "--lengths", "16,32,64,128",
             "--warmups", "2", "--repeats", "5",
             "--out", "results/lenovo_p2_cpu.json"],
        ]
    raise ValueError("Invalid mode")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("inspect", "p1", "p1text", "p2"), default="inspect")
    p.add_argument("--limit-seconds", type=int, default=240)
    a = p.parse_args()
    if not 10 <= a.limit_seconds <= 240:
        p.error("Trial limit must remain between 10 and 240 seconds")
    if not sys.platform.startswith("win"):
        p.error("Supervisor is intended for Windows")
    RESULTS.mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    manifest = {"mode": a.mode,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "runs": [], "safety": {
                    "max_gpu_c": 80, "max_gpu_mib": 3400,
                    "min_free_ram_gib": 8, "max_seconds": a.limit_seconds,
                    "sampling_interval_seconds": 2}}
    log_path = RESULTS / f"safe_runner_{a.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    print("START", a.mode, guard(), flush=True)
    error = None
    try:
        for i, args in enumerate(runs_for_mode(a.mode)):
            before = guard()
            print("RUN", i, args[1:], "state", before, flush=True)
            started = time.monotonic()
            measurements = [before]
            proc = subprocess.Popen(args, cwd=ROOT, env=env)
            reason = None
            try:
                while proc.poll() is None:
                    time.sleep(2)
                    try:
                        state = guard()
                        state["elapsed_seconds"] = round(time.monotonic() - started, 1)
                        measurements.append(state)
                        if len(measurements) % 5 == 0:
                            print("MONITOR", i, state, flush=True)
                        if time.monotonic() - started > a.limit_seconds:
                            raise RuntimeError("Per-run time limit reached")
                    except Exception as exc:
                        reason = str(exc)
                        if proc.poll() is None:
                            proc.kill()
                        break
                rc = proc.wait(timeout=10)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            record = {"index": i, "command": args[1:], "seconds": time.monotonic() - started,
                      "returncode": rc, "reason": reason, "telemetry": measurements}
            manifest["runs"].append(record)
            print("FINISHED", i, "returncode", rc, "reason", reason, flush=True)
            if reason or rc != 0:
                raise RuntimeError(reason or f"Child exited with code {rc}")
        print("COMPLETE", a.mode, "final", guard(), flush=True)
    except Exception as exc:
        error = str(exc)
        print("STOPPED", error, flush=True)
        raise
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = error
        log_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("SAFETY_LOG", log_path, flush=True)


if __name__ == "__main__":
    main()

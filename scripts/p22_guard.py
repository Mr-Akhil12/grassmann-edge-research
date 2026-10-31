"""P2.2 Windows local safety supervisor, independent of Remote Desktop.

Checks every ~2 seconds. No scheduling, no OS/driver/power changes.
Runs ONE bounded child; kills only its own child on any guard failure.
Outputs sampled telemetry & child logs under ignored results/p22/.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p22"


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def sensors():
    if sys.platform != "win32":
        raise RuntimeError("This guard is Windows-only")
    mem = MemoryStatus()
    mem.dwLength = ctypes.sizeof(MemoryStatus)
    power = PowerStatus()
    kernel = ctypes.windll.kernel32
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise RuntimeError("Cannot read physical RAM")
    if not kernel.GetSystemPowerStatus(ctypes.byref(power)):
        raise RuntimeError("Cannot read laptop AC power")
    cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw,clocks.current.sm",
           "--format=csv,noheader,nounits"]
    reading = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=6)
    line = reading.stdout.strip().splitlines()
    if len(line) != 1:
        raise RuntimeError("Unexpected NVIDIA GPU telemetry")
    vals = line[0].split(",")
    if len(vals) != 5:
        raise RuntimeError("Missing GPU telemetry")
    temp, vram, utilization, watts, clocks = (float(x.strip()) for x in vals)
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "gpu_temp_c": temp,
        "gpu_vram_mib": vram,
        "gpu_util_percent": utilization,
        "nvidia_power_w": watts,
        "sm_clock_mhz": clocks,
        "free_ram_gib": round(mem.ullAvailPhys / 2**30, 3),
        "ac_line": int(power.ACLineStatus),
        "battery_percent": int(power.BatteryLifePercent)
    }


def ensure_safe(t):
    # Guard even if telemetry abruptly missing. No attempts to ignore failure.
    if t["ac_line"] != 1:
        raise RuntimeError("AC adapter disconnected or unknown")
    if t["gpu_temp_c"] >= 78:
        raise RuntimeError("GPU temperature reached 78C abort threshold")
    if t["gpu_vram_mib"] >= 3000:
        raise RuntimeError("GPU VRAM used reached 3000MiB abort threshold")
    if t["free_ram_gib"] < 8:
        raise RuntimeError("Free RAM fell below 8GiB")
    return t


def run_child(args, seconds, label, *, cold_retest=False):
    OUT.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = OUT / (started_utc + "_" + label)
    manifest_file = prefix.with_suffix(".guard.json")
    out_path = prefix.with_suffix(".stdout.log")
    err_path = prefix.with_suffix(".stderr.log")
    telemetry = []
    manifest = {"run_label": label, "start_utc": started_utc,
                "command": args[1:], "max_seconds": seconds,
                "limits": {"temp_abort_c": 78, "total_vram_abort_mib": 3000,
                           "min_free_ram_gib": 8, "ac_required": True,
                           "child_cuda_allocator_fraction": 0.60,
                           "sensor_interval_s": 2},
                "samples": telemetry, "failure": None, "returncode": None,
                "cold_retest_authorized": cold_retest}
    t0 = time.monotonic()
    child = None
    try:
        if (OUT / "STOP_REQUESTED").exists() and not cold_retest:
            raise RuntimeError("STOP_REQUESTED marker exists")
        if (OUT / "ABORT_NOW").exists():
            raise RuntimeError("ABORT_NOW marker exists")
        before = ensure_safe(sensors())
        if cold_retest and before["gpu_temp_c"] > 38:
            raise RuntimeError("Cold retest requires GPU <=38C at launch")
        telemetry.append(before)
        print("P22 START", label, before, flush=True)
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr)
            while child.poll() is None:
                time.sleep(2)
                if (OUT / "ABORT_NOW").exists():
                    raise RuntimeError("ABORT_NOW requested by owner")
                if (OUT / "STOP_REQUESTED").exists() and not cold_retest:
                    raise RuntimeError("STOP_REQUESTED requested by owner")
                sample = ensure_safe(sensors())
                sample["elapsed_s"] = round(time.monotonic() - t0, 1)
                telemetry.append(sample)
                if len(telemetry) % 5 == 0:
                    print("P22 MONITOR", label, sample, flush=True)
                if time.monotonic() - t0 > seconds:
                    raise RuntimeError("Wall-clock child deadline exceeded")
            manifest["returncode"] = child.returncode
            if child.returncode:
                raise RuntimeError(f"Child failed exit={child.returncode}")
        print("P22 DONE", label, "s=", round(time.monotonic()-t0, 1), flush=True)
    except BaseException as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        print("P22 ABORT", label, manifest["failure"], flush=True)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
            manifest["returncode"] = child.returncode
        manifest["elapsed_s"] = round(time.monotonic()-t0, 2)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_file.write_text(json.dumps(manifest, indent=2)+"\n",encoding="utf-8")
        print("P22 LOGS", manifest_file, out_path, err_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inspect", "calibrate", "train"), default="inspect")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--kind", choices=("grassmann", "transformer", "local"), default="grassmann")
    parser.add_argument("--seed", type=int, choices=(11,17,23,31,47), default=11)
    parser.add_argument("--stop-step", type=int, default=50)
    parser.add_argument("--cold-retest", action="store_true",
                        help="One specific short seed-23 cold retry while STOP marker stays in place")
    p = parser.parse_args()
    if not (30 <= p.seconds <= 240):
        parser.error("Only bounded 30–240 s supervised probes")
    if p.cold_retest and not (
        p.mode=="train" and p.kind=="grassmann" and p.seed==23 and
        p.stop_step==100 and (p.dim,p.layers,p.rank,p.batch,p.context)==(288,6,28,8,512) and
        p.seconds<=120 and (OUT / "STOP_REQUESTED").is_file()):
        parser.error("Cold retest only permitted for interrupted Grassmann seed23 40->100")
    if p.mode == "inspect":
        cmd = [sys.executable, "-m", "pytest", "-q"]
        label = "inspect"
    elif p.mode == "train":
        if (p.dim,p.layers,p.rank,p.batch,p.context) != (288,6,28,8,512):
            parser.error("Training must use locked calibrated configuration")
        if not 10 <= p.stop_step <= 2000:
            parser.error("Invalid training step")
        label = f"train_{p.kind}_seed{p.seed}_stop{p.stop_step}"
        cmd = [sys.executable, "-m", "scripts.p22_train",
               "--kind", p.kind, "--seed", str(p.seed),
               "--stop-step", str(p.stop_step),
               "--checkpoint-every", "20"]
    else:
        label = f"cal_d{p.dim}_l{p.layers}_r{p.rank}_b{p.batch}_c{p.context}"
        cmd = [sys.executable, "-m", "scripts.p22_calibrate",
               "--dim", str(p.dim), "--layers", str(p.layers),
               "--rank", str(p.rank), "--batch", str(p.batch),
               "--context", str(p.context), "--steps", "3",
               "--out", "results/p22/" + label + ".json"]
    run_child(cmd, p.seconds, label, cold_retest=p.cold_retest)


if __name__ == "__main__":
    main()

"""Read-only host inventory. No usernames, serials, IPs, secrets or file scanning."""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def safe(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    git_exe = shutil.which("git")
    if git_exe is None and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe"
        if candidate.is_file():
            git_exe = str(candidate)
    info = {"os": platform.system(), "release": platform.release(),
            "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
            "free_disk_gib": round(shutil.disk_usage(Path.cwd()).free / (1024**3), 2),
            "git": safe([git_exe, "--version"]) if git_exe else None,
            "nvidia": safe(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,temperature.gpu,driver_version",
                "--format=csv,noheader"]),
            "adb_available": shutil.which("adb") is not None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 2)
    except ImportError:
        info["torch"] = None
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

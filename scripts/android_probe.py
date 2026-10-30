"""Read-only adb inventory; does not read personal files or device identifiers."""
import json
import shutil
import subprocess


def adb(args):
    try:
        return subprocess.check_output(["adb", "shell", *args], text=True,
                                       stderr=subprocess.DEVNULL, timeout=8).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    if not shutil.which("adb"):
        print("Android Platform Tools unavailable; phone untouched.")
        return
    fields = {
        "model": ["getprop", "ro.product.model"],
        "android_release": ["getprop", "ro.build.version.release"],
        "sdk": ["getprop", "ro.build.version.sdk"],
        "abi": ["getprop", "ro.product.cpu.abi"],
        "soc_model": ["getprop", "ro.soc.model"],
        "memory_info": ["sh", "-c", "grep MemTotal /proc/meminfo"],
        "battery_state": ["sh", "-c", "dumpsys battery | grep -E 'level:|temperature:|status:'"],
    }
    print(json.dumps({k: adb(v) for k, v in fields.items()}, indent=2))


if __name__ == "__main__":
    main()

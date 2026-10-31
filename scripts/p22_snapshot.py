"""Publish-auditable P2.2 JSON/telemetry snapshot, excluding model checkpoints.

Copies only listed data formats, verifies no personal paths/credentials,
and generates SHA256 manifest for independent research auditing.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results"/"p22"
OUT=ROOT/"experiments"/"2026-10-31-p22-scaled"
SOURCES={
    "calibration":["cal_d*.json"],
    "training":["train/*.metrics.json"],
    "telemetry":["*.guard.json"],
    "campaign":["campaign_*.json"],
    "failure_logs":["*train_grassmann_seed23_stop100.stdout.log"]
}
BAD=("C:\\Users\\", "/Users/", "ghp_", "gho_", "sk-")
def main():
    files=[]
    for group,patterns in SOURCES.items():
        target=OUT/group
        target.mkdir(parents=True,exist_ok=True)
        for pattern in patterns:
            for src in sorted(BASE.glob(pattern)):
                if not src.is_file():
                    continue
                if src.suffix not in (".json",".log"):
                    raise ValueError(f"Unexpected suffix {src.name}")
                payload=src.read_bytes()
                # Git tracks these JSON/text files with LF endings on all OSes.
                # Preserve all content but canonicalize CRLF before hashing.
                publish_payload=payload.replace(b"\r\n", b"\n")
                # Whole record is intentionally inspectable; do not copy
                # sensitive paths/credentials even if a future logger changes.
                decoded=publish_payload.decode("utf-8",errors="strict")
                if any(s in decoded for s in BAD) or re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", decoded):
                    raise ValueError(f"Sensitive string in {src.name}")
                # .log files are intentionally ignored by Git; export as .txt
                # so failed-run evidence remains publicly trackable.
                dst=target/(src.stem + ".txt" if src.suffix==".log" else src.name)
                if dst.exists() and dst.read_bytes().replace(b"\r\n", b"\n")!=publish_payload:
                    raise ValueError(f"Refusing changed already-published file {dst}")
                dst.write_bytes(publish_payload)
                files.append({
                    "relative_path":dst.relative_to(ROOT).as_posix(),
                    "bytes":len(publish_payload),
                    "sha256":hashlib.sha256(publish_payload).hexdigest(),
                    "source_sha256":hashlib.sha256(payload).hexdigest()
                })
    manifest={
        "status":"PARTIAL SCALED P2.2 ENDURANCE PILOT; STOPPED ON 78 C GPU LIMIT",
        "captured_utc":datetime.now(timezone.utc).isoformat(),
        "excludes":"Private 75MB model and optimizer checkpoints remain under ignored results/",
        "files":files,
        "note":"Truncated Grassmann seed23 has checkpoint at step40 but no completed 100-step metrics; partial stdout and guard failure are preserved."
    }
    (OUT/"SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("SNAPSHOT_FILES",len(files),"BYTES",sum(f["bytes"] for f in files),
          "DEST",OUT,flush=True)

if __name__=="__main__":
    main()

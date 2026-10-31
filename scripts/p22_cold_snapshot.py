"""Snapshot a single allowed cold-start retry without private optimizer files."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

root=Path(__file__).resolve().parents[1]
src=root/"results"/"p22"
dst=root/"experiments"/"2026-10-31-p22-cold-retest"
dst.mkdir(parents=True,exist_ok=True)
sources={
 "guard.json":src/"20261031T063729Z_train_grassmann_seed23_stop100.guard.json",
 "training_metrics.json":src/"train"/"grassmann_seed23_d288_ctx512.metrics.json",
 "training_stdout.txt":src/"20261031T063729Z_train_grassmann_seed23_stop100.stdout.log",
}
blocked=("C:\\Users\\","/Users/","ghp_","gho_","sk-")
files=[]
for dest,source in sources.items():
    content=source.read_bytes().replace(b"\r\n",b"\n")
    s=content.decode("utf-8")
    if any(term in s for term in blocked):
        raise RuntimeError(f"Sensitive marker in {source.name}")
    out=dst/dest
    if out.is_file() and out.read_bytes()!=content:
        raise RuntimeError(f"Refusing changed published data {out}")
    out.write_bytes(content)
    files.append({"file":dest,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
assert json.loads((dst/"guard.json").read_text())["failure"] is None
assert json.loads((dst/"training_metrics.json").read_text())["steps_completed"]==100
(dst/"MANIFEST.json").write_text(json.dumps({
 "status":"successful guarded cold-start retest; original thermal abort retained unchanged",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "source_revision_before_retest":"6defb0180f1df87b946a58c83ce91c20ea9666c7",
 "source_checkpoint":"grassmann seed23 step40 original SHA256 BC0F67B1D1CFD21F972ED7D8A03E73F205ACAE5FF4029E7D92DEFBD2ABCA931F",
 "checkpoint_binary_published":False,
 "files":files
},indent=2)+"\n",encoding="utf-8")
print("RETEST_SNAPSHOT_OK",files)

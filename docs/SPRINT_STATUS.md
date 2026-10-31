# P0–P2 sprint status — 31 October 2026

The public project remains research-first and reproducible. **No claim that
Grassmann beats attention has been made. No Xiaomi tests were attempted.**

## Verified achievements

- Mathematical paper-literal (future pairing) and causally corrected (past
  pairing) operator variants committed.
- Independent scalar double-precision Pluecker reference, gradient and
  prefix invariance tests. Causal operator matches an incremental rank-state
  reference within tested numerical tolerances.
- Transformer and local-mixer incremental references have full-forward
  parity tests; validated on fresh Linux and Windows GitHub Actions runners.
- Parameter counts and training curves retained in a common format.
- Exploratory two-seed synthetic pilot and untrained CPU prefill/decode
  repetitions run on Windows and Linux; **raw JSON committed** under
  experiments/2026-10-31-crossplatform-ci/.
- Pinned, SHA-256 checked WikiText-2-raw-v1 preparation code published,
  but full natural-text training has NOT been executed.
- A guarded, intentionally bounded Windows GPU runner has been prepared
  under scripts/run_lenovo_safely.ps1. **Not yet exercised on the Lenovo.**

## Important outstanding acceptance criteria

### P0
Formal causality conflict can be demonstrated at model level, but original
author intent, paper-literal training alignment, ambiguous normalization
and reproduction of 13–18M parameter reported models remain unverified.

### P1
Natural-language experiments, fully parameter/FLOP-matched models,
optimizer/tuning-equivalent baselines, independent multi-seed significance
and paper-scale WikiText-2 reproduction are NOT complete. CI synthetic
experiments have only 12 optimization steps with two seeds.

### P2
Linux and Windows CI CPU reference inference was exercised and raw timings
retained. The Lenovo RTX 3050 GPU full benchmarking suite has NOT run.
No thermal or power results on the actual Lenovo have been measured.
No optimized fused geometry kernels or calibrated energy telemetry exist.

## Operational block

On 2026-10-31 the Remote Desktop Commander connection to Akhil-PC
stopped responding, subsequently appearing **offline**. No GPU trials were
launched during the communication failure; no detached background jobs were
scheduled. The repo was advanced through the authenticated Mac solely for
source editing, publication and GitHub Actions operations.

The authoritative public repository is
https://github.com/Mr-Akhil12/grassmann-edge-research
The Lenovo's local checkout may be BEHIND public main. Before any testing
from Windows, verify that working tree is clean and fast-forward it from the
public remote; do not force-reset or overwrite local work.

## Safe remaining P2 execution

Only after a fresh successful read-only Lenovo hardware audit and clean
Git sync from the public repository:
1. Inspect free RAM, SSD, AC, GPU temperature, VRAM and driver.
2. Activate existing per-project .venv and install any declared new deps
   inside .venv only; do not upgrade drivers.
3. Run guarded `-Mode inspect`; verify all tests.
4. Run guarded `-Mode p1` exploratory synthetic lag=4/lag=24, preserving
   every raw sample; monitor after each child.
5. Run guarded `-Mode p2` bounded GPU inference profiles.
6. Publish raw per-run results, observation ranges and failures.
7. Keep issue checkboxes/phase gates unmarked until the acceptance criteria
   have actually been met.

P3 requires the user's Xiaomi 15T and explicit device access approval;
no Android debugging has been attempted.

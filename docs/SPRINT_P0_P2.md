# P0–P2 controlled research protocol — 31 October 2026

This is an **open scientific investigation**, not a claim that geometric
Grassmann flows outperform causal attention. No Xiaomi/Android access occurs
until P3 with owner involvement.

## P0 — mathematical correctness

- Examine the source paper, [arXiv:2512.19428](https://arxiv.org/abs/2512.19428),
  especially the literal forward pairing t+delta and causal-language claim.
- Implement a paper-literal right-looking variant separately from a corrected
  past-only left-looking variant. Never quietly conflate them.
- Verify Pluecker exterior products against a double-precision scalar oracle.
- Require prefix invariance and zero future-to-past gradients for causal
  methods; deliberately show the literal variant violating the invariant.
- Verify full-versus-incremental equivalence at multiple sequence lengths.
- Do not claim original-author intent or exact experimental reproduction
  without further evidence.

## P1 — matched-task exploratory training

### Synthetic delayed-copy

- Causal Grassmann, causal Transformer, non-geometric local-mixing comparator.
- Seeds 11,17,23; model order fixed before evaluation.
- Context 64, dimension 32, 2 layers, rank 8, offsets {1,2,4,8}.
- Batch 4; 60 steps per seed and model; AdamW 0.002.
- Separate lag-4 and lag-24 tasks; held-out deterministic validation.
- Report all loss traces, run durations, parameter counts and variation.

### Real text

- WikiText-2 raw, immutable upstream revision
  6231e49f19a707241d6b84d9cff60a3a86b85a85.
- SHA-256 verified train and validation; test not downloaded.
- UTF-8 byte vocabulary (256 values), NOT the WordPiece vocab of the paper.
- Same three model classes and three seeds; 60 steps; context 64; batch 4.
- Validation bits/byte reported alongside per-seed values and deviations.
- Models have different parameter counts and not equalized FLOPs; no
  confirmatory language-quality claim can be made from these tiny runs.

## P2 — bounded reference inference

- Compare untrained full-context prefill **separately** from cached
  incremental decoding. The reference cached mode uses Python token loops,
  not a fused production kernel.
- Validate numerical parity before timing.
- Test lengths 16,32,64,128 on both Lenovo CPU and RTX 3050.
- Record all raw times (five repetitions following two warmups), medians,
  PyTorch maximum allocated CUDA memory and theoretical cache payloads.
- This is kernel/implementation diagnostic evidence, not an optimized
  quality-versus-speed comparison.

## Hardware safety

- Keep all software and model tests within isolated project virtualenv.
- No firmware, BIOS, drivers, Windows execution policy, antivirus,
  GPU clocks, voltage, power or unrelated project changes.
- Python safety supervisor starts one child experiment at a time and checks
  temperature, VRAM, free RAM and runtime every ~2 seconds.
- Hard limits: temp below 80°C, used VRAM below 3400 MiB, free RAM at
  least 8 GiB, per-run duration no more than 240 seconds.
- Child receives an additional 60% PyTorch CUDA memory fraction limit.
- Supervisor stops its *own* child if anything fails; no detached jobs.
- Telemetry is sampled, so it does not establish unsampled peak behavior.

### Verified Windows commands

After checking the Lenovo is powered and the Git checkout clean, run:

```text
.venv/Scripts/python.exe -m scripts.safe_windows_runner --mode inspect
.venv/Scripts/python.exe -m scripts.safe_windows_runner --mode p1
.venv/Scripts/python.exe -m scripts.safe_windows_runner --mode p1text
.venv/Scripts/python.exe -m scripts.safe_windows_runner --mode p2
```

The earlier PowerShell launcher was *not* used, because the Lenovo's
default PowerShell policy disallows scripts. No security policy was changed.

## Evidence levels and outstanding gates

- **A — correctness:** numerical/causality/cached-parity tests verified.
- **B — exploratory hardware:** raw guarded Lenovo GPU/CPU pilots completed.
- **C — scientific quality:** not yet established; requires substantial,
  truly parameter/FLOP-matched training, context and compute ablations,
  suitable dataset controls, and independent multi-seed replication.
- **D — edge energy:** not attempted; user authorization required for P3.

Raw Lenovo measurements and all limitations are recorded in
[experiments/2026-10-31-lenovo-rtx3050](../experiments/2026-10-31-lenovo-rtx3050/README.md).
Preserve every unfavorable and failed run; do not promote exploratory
numbers to paper-scale claims.

# P0–P2 controlled research sprint — public protocol

**Scope:** software correctness, rigorous experimental infrastructure, and
bounded inference evaluation on Windows/RTX 3050 when reachable.
**Explicitly out of scope:** Xiaomi 15T access, Android/ADB, external machine
administration, firmware/drivers, unsupported claims about breakthroughs.

This document describes a *research program*, not an assertion of completed
paper reproduction. Phase status is only advanced by verified evidence.

## P0: mathematical specification and causality

Source: Zhang Chong, arXiv:2512.19428v1, §3.2.
The manuscript explicitly uses the pair (z_t, z_(t+delta)) while calling
it causal. This is inconsistent with standard prefix-limited next-token
prediction at output position t.

We implement `paper_literal` as written, and `past_causal` replacing
t+delta with t-delta. The wedge signs depend on vector order; keep
the ordering explicit. See docs/P0_CAUSALITY_AUDIT.md.

Acceptance gates: scalar double-precision Pluecker oracle, numerical and
gradient tests, prefix invariance, an actual noncausal diagnostic, full-vs-
incremental parity. External author intent and exact 13M model outcome
remain unverified; avoid claiming full paper reproduction.

## P1: controlled exploratory learning

**Primary negative control:** non-geometric local pair mixing with identical
local offset access. Comparator: standard full causal attention Transformer.
Main hypothesis: Grassmann could improve a quality/compute tradeoff beyond
a simpler local mixer. Null: any benefit disappears under local baseline or
optimized attention.

Predeclared pilot configuration:
- Input: synthetic autoregressive delayed-copy sequence (deterministic).
- Seeds: 11, 17, 23; train sequence regenerated for each step.
- Two separate dependency lags: 4 and 24, with 64-token context.
- Rank=8, d=32, two layers, offsets {1,2,4,8}, vocab=32, batch=4.
- 60 optimization steps/model/seed; AdamW lr=0.002, weight decay 0.01.
- Evaluation uses 4 held-out batches derived from distinct seed domain.
- Model order: causal Grassmann, causal Transformer, non-geometric local.
- Record all train losses, parameter counts, validation losses, run durations,
  exact CLI arguments, seeds and PyTorch/Git revision.
- No tuning, early stopping or validation-based hyperparameter selection.

**This is only a small exploratory pilot.** It is not enough to compare
language-model quality, estimate a robust performance distribution, or reproduce
WikiText-2 results. The existing architectures are NOT precisely parameter-
or FLOP-matched. Relative parameter mismatch is reported, not concealed.
A confirmatory paper-scale investigation needs data provenance, tokenizer
alignment, size-matched runs, tuning budget and sufficient independent seeds.

## P2: bounded inference and caching

- Confirm each architecture's full-sequence output matches true incremental
  *past-only* decoding. A paper-literal future-looking model must reject
  such a decode.
- Record batch=1, float32, random-init weights; d=32, two layers, rank=8,
  context lengths 16, 32, 64, 128.
- Separate full-sequence prefill and token-by-token cached decoding.
- Warm up twice, take five timed repetitions, synchronize CUDA each time.
- Capture all raw samples, median/min/max and PyTorch max allocated VRAM.
- Report cache *payload* estimates separately from actual GPU allocation.
- Guard 4GB VRAM GPU using per-process PyTorch allocation fraction <=60%.
- Windows monitor enforces temp <80 C, used VRAM <3400MiB, free RAM >=8GiB,
  and max runtime per trial <=240s; aborts its OWN child process.
- No overclocking, GPU driver changes, BIOS changes, power tweaks or antivirus
  disabling. Use A/C power.
- No energy-per-token claim without calibrated energy instrumentation.
- Do NOT interpret untrained performance as generative AI model quality, and
  do NOT treat Python-loop cached decode as an optimized device kernel.

## Commands

After `python -m pip install -e ".[dev]"`:

```bash
python -m pytest -q
python -m scripts.controlled_synthetic --device cpu --steps 12 --seeds 11,17 --out results/controlled_ci.json
python -m scripts.benchmark_p2 --device cpu --lengths 16,32 --warmups 1 --repeats 3 --out results/p2_ci.json
```

On a **verified, powered Lenovo** from Windows PowerShell:
```powershell
powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode inspect
powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode p1
powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode p2
```

The default is *inspect*, which runs no GPU training. Each other mode
requires explicit invocation. This is **not** a scheduled/background job.

## Evidence grades

A. Formal code/numerical invariants with independent test execution.
B. Engineering CPU/GPU performance smoke test (reproducible raw measurements).
C. Matched multi-seed scientific model-quality result (not yet achieved).
D. Calibrated edge-device energy result (requires P3, not attempted).

Do not upgrade evidence category without additional experiments. A broken
connection to the Lenovo means P2 GPU measurements remain PENDING, irrespective
of GitHub CI status.

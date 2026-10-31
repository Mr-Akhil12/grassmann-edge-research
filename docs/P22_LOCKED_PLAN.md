# P2.2 locked protocol — Lenovo scaled endurance study (2026-10-31)

**Status: calibrated, short rehearsals completed; extended campaign HALTED at the 78°C GPU guard. No multihour run authorized.** See [public raw evidence and incomplete-trial report](../experiments/2026-10-31-p22-scaled/README.md). P2.2 is paused pending cooling review.
Purpose: establish defensible **small-scale, implementation- and hardware-
specific** results, not a universal Grassmann-vs-attention verdict.

## Research hypotheses and stop criteria

H0: after controlling model size and training budget, causal Grassmann
does not reliably improve validation loss / bits-per-byte relative to a
simpler causal local mixer. Compare separately to full-causal attention.
H1: there is reproducible gain in quality/compute/VRAM tradeoff beyond
the non-geometric local control under the predeclared constraints.

**Primary metric**: held-out WikiText-2 raw UTF-8 **bits per byte** at fixed
training tokens. No test split access or tuning on the test set.
**Secondary**: median CUDA-synchronized training step, tokens/s, allocated
and reserved CUDA memory, prefill vs cached-decode latency and sustained
clock/thermal stability. NVIDIA wattage telemetry is not calibrated
joules/token.

## Calibration results (RTX3050 Laptop 4GB, FP32, 3 training steps/model)

| d / layers / rank / B / context | Params mismatch | Grassmann alloc/reserve MiB | Transformer alloc MiB | Local alloc MiB |
|---|---:|---:|---:|---:|
| 128/4/16/4/256 | 0.054% | 135.53 / 152 | 67.76 | 74.46 |
| 192/4/24/8/256 | 0.038% | 465.54 / 500 | 152.86 | 176.77 |
| 256/6/24/8/512 | 0.029% | 1464.17 / 1704 | 494.41 | 596.71 |
| **288/6/28/8/512** | **0.026%** | **1885.08 / 2124** | **572.35** | **672.67** |

**Decision (frozen): d=288, 6 layers, rank=28, B=8, context=512,
six causal offsets (1,2,4,8,16,32), FP32; no AMP and no gradient
checkpointing in confirmatory series.** All architectures use the
same embedding width and depth. Their FFN intermediate widths are
individually matched to the Transformer reference's parameter count.
This does *not* equalize FLOPs, activation RAM, receptive field or kernel
maturity. Width matching is predeclared, not outcome-based.

The corrected Grassmann and non-geometric local mixer only have a
theoretical maximum receptive-field span of 1+6*32=193 positions,
whereas full causal Transformer attends to 512. This limits any
quality equivalence claim.

The 1.84GiB Grassmann peak is **allocated PyTorch tensor memory**;
the 2.07GiB number is allocator reserved memory. Neither is a reliable
proxy for sustained GPU utilization. A short calibration cannot prove
that every extended step remains within the bound. Measured CUDA
allocator cap stays at 60% of 4096MiB (<2.46GiB); hardware guard
aborts at 3,000MiB observed total VRAM.

## Reproducibility / training configuration

- Corpus: pinned Salesforce/wikitext `wikitext-2-raw-v1`, revision
  `6231e49f19a707241d6b84d9cff60a3a86b85a85`.
- Downloaded train and validation SHA256-checked before use.
- Vocabulary: raw UTF-8 bytes (256 values). No WordPiece comparison.
- Model kinds: `grassmann`, `transformer`, `local`.
- Predeclared independent seeds: `11,17,23,31,47`.
- Same seed-specific pseudo-random train-window generator in each model.
- Each sample includes 512 bytes; next-byte cross-entropy.
- Batch 8, sequence 512, context length 512; 4096 input-byte
  positions processed per optimizer step, approximate next-byte target
  count 4088 per step due shifted labels.
- Optimizer: AdamW lr=0.0003, weight decay 0.01, gradient clipping 1.
- Initial target: **1000 optimizer steps per architecture/seed**,
  4,096,000 input bytes processed per run; 15 runs and a total
  61,440,000 input-byte positions across all models.
- Validation: 8 fixed-seed windows per run, train/validation splits
  separate. Same validation-window generator per seed/model.
- Preserve all losses, per-step results, checkpoint SHA256, environment,
  tensor allocation peaks and failed runs; never cherry-pick seeds.
- At most 100-step resumable training blocks per subprocess; checkpoint
  every 20 steps to ignored local results before next block.
- Must complete an **initial 50-step seed-11 triplet** and CPU/GPU
  correctness tests before launching the longer five-seed series.
- Test split held back; no test-based hyperparameter tuning.
- If 1000-step model loss shows nonconvergence, report that limitation,
  not a false conclusion. Further phase explicitly requires a separately
  pre-registered matched-compute extension.

## Endurance / operational control

- Only one GPU child process at a time, always via a Windows-native
  Python supervisor. No OS service, task scheduler or firmware changes.
- Supervise using live NVIDIA temperature, VRAM, GPU utilization, clock,
  indicated power and Windows AC/RAM. All telemetry sampled and saved
  about every 2 seconds; sensor failures are hard stops.
- **Abort** when AC power is missing/unknown, GPU temp >=78°C,
  NVIDIA-reported used VRAM >=3,000MiB, free system RAM <8GiB,
  or 240-second per-child wall-clock deadline. GPU PyTorch allocator
  capped at 60% of total. Stop on CUDA errors, non-finite gradients,
  rising allocation or unexplained throttling.
- AC, laptop cooling/airflow and unrelated GPU process conflicts must
  remain verified. Device is user's everyday laptop; halt rather than
  seek higher utilization through unsafe conditions.
- Independent checkpoints can resume after interrupted bounded blocks.
  Never auto-restart a failed child blindly; preserve forensic logs.
- A long session cannot be personally monitored continuously through a
  normal chat; local supervision must function even on RDC disconnect.
- Daily/session maximum should be reviewed with user before any
  unattended multihour campaign. **No 24-hour burn-in is authorized
  by this protocol alone.**

## Intended experiment stages and release gates

**Gate A (now):** 46+ correctness tests, parameter matching, guarded
calibration, and seed-11 50-step triplet. Inspect full telemetry.
**Gate B:** five-seed 1000-step train/validation campaign, round-robin
architecture ordering to reduce thermal/time-of-day confounding,
committed raw metrics and failure records.
**Gate C:** inference on actual trained, quality-characterized
checkpoints at lengths 64/128/256/512; separate prefill, single-step
cached decode and 512-token cached sweep with repeated timings.
**Gate D:** repeat under cooling/clock controls, look for trends in
latency, memory and throttling; report p50/p95 and variability.
**Gate E:** publish negative/positive/null results and statistical
uncertainty, disclose kernels, non-equivalent receptive fields and
calibration selection. Require independent replication before broader
claims.

No Xiaomi 15T or Android activity occurs before P3 and user authorization.

## How to run manually (not a background scheduler)

From the repository root using its Windows project virtualenv:
`.` followed by `venv/Scripts/python.exe -m scripts.p22_guard`.
Modes: `--mode inspect`, `--mode calibrate --dim 288 --layers 6
--rank 28 --context 512 --batch 8 --seconds 150`,
or `--mode train --dim 288 --layers 6 --rank 28 --context 512 --batch 8
--kind grassmann --seed 11 --stop-step 50 --seconds 150`.
Only run under the Python guard. Results and checkpoints remain in
ignored `results/p22` until explicitly audited for public release.

See GitHub issue #7 for independent review and outstanding gates.

# P2.2 scaled GPU calibration and interrupted endurance trial — 31 October 2026

## Status: STOPPED AT THERMAL GUARD; NOT AN ALL-DAY VALIDATION

The 4 GB RTX 3050 laptop successfully supported short training trials
of a ~6.29M-parameter causal Grassmann LM. However, sequential
higher-load trials accumulated heat and the **78°C predeclared GPU
abort threshold triggered during Grassmann seed 23**.
The local Python supervisor killed only its own model process and the
outer campaign immediately exited. No further training was launched.

**This safety intervention is a result, not a run to omit.**
A local `results/p22/STOP_REQUESTED` marker now prevents accidental
restart until the user checks ventilation and the thermal profile is
reassessed.

## Methods (predeclared before observing endurance results)

Hardware: Lenovo IdeaPad Gaming 3 15IHU6, i5-11320H, 32GB RAM,
GeForce RTX3050 Laptop 4GB, Windows 11, NVIDIA 595.79,
PyTorch 2.10.0+cu128, FP32.

Training: WikiText-2 raw train/validation, pinned upstream revision
`6231e49f19a707241d6b84d9cff60a3a86b85a85`, byte tokenizer (256),
context 512, batch 8, six layers, hidden size 288, Grassmann rank 28,
causal offsets (1,2,4,8,16,32). AdamW lr0.0003, wd0.01, clipping 1.
Validation used the held-out validation split, 8 predeclared batches/seed.
Test split was not accessed.

FFN widths were adjusted to match trainable parameter counts, preserving
common embedding width/layers, though not identical FLOPs or receptive
field. Actual params:
- Grassmann: 6,290,604
- Causal Transformer: 6,289,920
- Non-geometric local: 6,291,570
- Maximum parameter spread: 0.026%.

Local-mixer and Grassmann receptive-field span is 193 tokens; full
attention can attend to the entire 512-token prefix. This difference
is a significant quality-comparison limitation.

## Calibration

Each row: 3 optimizer steps/model on actual hardware, CUDA peak tensor
allocation and allocator reserved (MiB). Not an endurance test.

| d / layers / rank / batch / ctx | Grass alloc/reserve | Transformer alloc | Local alloc |
|---|---:|---:|---:|
| 128 / 4 / 16 / 4 / 256 | 135.53 / 152 | 67.76 | 74.46 |
| 192 / 4 / 24 / 8 / 256 | 465.54 / 500 | 152.86 | 176.77 |
| 256 / 6 / 24 / 8 / 512 | 1464.17 / 1704 | 494.41 | 596.71 |
| **288 / 6 / 28 / 8 / 512** | **1885.08 / 2124** | **572.35** | **672.67** |

The target allocates ~1.84GiB of *live PyTorch tensors* and reserves
~2.07GiB through PyTorch's caching allocator. During active training,
NVIDIA telemetry sampled about 2,215MiB total VRAM use.
These measurement categories differ; do not confuse them.

## Finite checkpoint/recovery trial

A 50-step seed-11 triplet completed on all architectures.
The Grassmann seed-11 checkpoint then successfully **resumed from step
50 to 100**, retaining the same dataset and stochastic state.
Checkpoint files are ~75MiB each and remain in ignored local
`results/p22/train`; only their hashes and measurement summaries
are publicly shared.

The subsequent planned seed-11/17/23, 100-step-per-model campaign
was **partially completed**, then halted on thermal guard.

| Seed | Grassmann | Transformer | Local mixer |
|---|---:|---:|---:|
| 11 | 3.58968 | 3.63949 | 3.54147 |
| 17 | 3.57842 | 3.63172 | 3.54033 |
| 23 | **INCOMPLETE** | **NOT RUN** | 3.53223 |

Values are validation bits-per-byte at *100 steps*; lower is better.
The completed models each processed 409,600 input-byte positions.
These are very early, undertrained scores; they are **not** enough for
any confirmatory architecture ranking.

Grassmann seed 23 saved a recoverable checkpoint at **step 40** before
the thermal stop; it has **no valid 100-step validation score**. The
partial stdout and the complete guard record are published; the raw
checkpoint is held locally and has been independently inspected at
step 40. There are seven completed 100-step model/seed records.

## Safety telemetry and failure

- Independent Windows-native watchdog checked AC, GPU temperature,
  NVIDIA VRAM use, GPU utilization, GPU wattage, clock and free RAM
  roughly every two seconds.
- Limits: abort at >=78°C GPU, >=3000MiB NVIDIA VRAM used,
  <8GiB free RAM, missing AC or telemetry, or 180s per child.
- PyTorch CUDA allocator limited to 60% of 4GB.
- Guard worked as designed: **at least one sensor observation was
  >=78°C**, triggering the explicit exception and killing its child.
- Last recorded non-triggering sample before the abort: **77°C**,
  2,215MiB VRAM, 99% GPU use, ~70.82W NVIDIA-reported GPU power.
  The threshold-triggering reading was not appended to the normal
  telemetry list because the guard raised immediately; the failure
  field records the crossing.
- Earlier recorded seed-17 Grassmann block reached around 72°C.
- GPU temperature dropped to 62°C and subsequently 52°C with
  0MiB VRAM allocated, idle GPU and **no Python training processes**.
- AC remained connected; no observed OOM.
- No driver/BIOS/clock/power/Windows execution policy changes.
- No CPU package temperature instrumentation was available, so
  full-system thermal sustainability is **not established**.
- No calibrated energy or battery consumption per token was measured.

## Conclusions we can and cannot draw

**Supported:** a 6.29M parameter-matched reference model trio works,
checkpoint/resumption works, live Grassmann tensors occupy ~1.84GiB,
and short-run validation loss is finite. Sustained back-to-back
operation at the locked 2GB-scale on this laptop **hit the configured
thermal stop**. The independent safety guard functioned correctly.

**Not supported:** that Grassmann beats a Transformer or the local
control, that this is a faithful WordPiece paper reproduction,
that the GPU would have suffered damage at 78°C, or that hours-long
unattended training is safe under current ventilation/ambient conditions.

## Next-step decision gate

Do not resume automatically. Owner should verify unobstructed laptop
intake/exhaust, elevated rear/cooling stand, stable AC and acceptable
ambient temperature. Consider physical cooling improvements first.
Then pre-register a revised P2.2b configuration/thermal protocol:
possibly the previously calibrated 256d/6-layer/24-rank context-512
candidate (Grassmann ~1.46GiB peak tensor allocation), shorter
training blocks with cooldown until GPU <45°C, and an earlier
thermal warning threshold. Reducing memory alone may not reduce
GPU power substantially; **recalibrate temperatures** before
campaign continuation.

Results, including negative findings, remain publicly auditable.
Only the user-authorized Xiaomi 15T is permitted at P3; P3 has
not started.

## Data files

This directory contains raw experiment values copied from the ignored
`results/p22/` run directory. Text line endings were canonicalized to
LF for cross-platform Git and checksum verification; numerical values
were not changed. `SNAPSHOT_MANIFEST.json` retains both source-byte and
published-file SHA-256 digests. Model/optimizer checkpoint binaries were
deliberately excluded. No private identifiers, tokens or personal
filesystem paths were copied.

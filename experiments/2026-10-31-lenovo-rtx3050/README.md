# Lenovo RTX 3050: guarded P1/P2 sprint — 31 October 2026

## Evidence status

**Executed on the Lenovo**: three-seed synthetic pilots, a three-seed
byte-level WikiText-2 pilot, and repeat-run untrained CPU/CUDA inference
profiles. None of these experiments is an independent reproduction of the
paper's original 13–18M parameter WordPiece WikiText-2 comparisons. No
energy-per-token, Android, power-efficiency or optimized-kernel advantage
is established.

The rig: Intel i5-11320H (4C/8T), 32GB RAM, NVIDIA RTX 3050 4GB,
Windows 11, NVIDIA driver 595.79, Python 3.11.9,
PyTorch 2.10.0+cu128. The supervisor sampled environmental telemetry
every approximately two seconds and terminated only its own child jobs
when necessary. No safety limit triggered, no jobs were left running,
and no driver, clock, BIOS, power or Windows security setting was changed.

## 1. P1 synthetic delayed-copy pilots

- Models: past-only Grassmann, causal Transformer, non-geometric local.
- Three seeds: 11, 17, 23; sequence length 64; d=32; 2 layers; rank=8.
- 60 training steps, batch 4, AdamW, same training-token budget per seed.
- Different parameter counts; maximum mismatch **9.04%**.
- Held-out deterministic validation, cross-entropy in nats; lower is better.

| Task lag | Grassmann mean ± SD | Transformer mean ± SD | Local mean ± SD |
|---|---:|---:|---:|
| 4 | 3.54188 ± 0.01724 | **3.43020 ± 0.04885** | 3.53323 ± 0.03844 |
| 24 | 3.50342 ± 0.01095 | 3.48581 ± 0.00661 | **3.48526 ± 0.01036** |

These runs are undertrained and do not establish statistical superiority.
At lag 24, the two-layer Grassmann theoretical receptive field is
1 + 2*max(1,2,4,8) = 17 positions, smaller than the desired lag.
Because model construction and target dynamics differ, this is a useful
structural limitation, not a publication-grade long-range evaluation.

## 2. P1 real-text WikiText-2 byte-level pilot

- Dataset: Salesforce/wikitext, wikitext-2-raw-v1,
  pinned revision `6231e49f19a707241d6b84d9cff60a3a86b85a85`.
- Training split upstream SHA256:
  `49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701`.
- Validation split upstream SHA256:
  `a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f`.
- The pinned initial script had the **wrong validation/test hashes**
  copied from the dataset's other revision. The verifier rejected the
  validation file; immutable upstream metadata was checked and the
  correction is included in this commit. Train/val now pass SHA256 checks.
- Test split **never opened or used**.
- Tokenizer: raw UTF-8 bytes, fixed vocab 256; this is **not** the
  paper's WordPiece benchmark, so do not compare perplexities.
- Context 64, dim 32, layers 2, rank 8, batch 4, seeds 11/17/23.
- 60 AdamW optimizer steps per seed/model, 15,360 train tokens/run.
- Parameter counts: Grassmann 42,000; Transformer 43,904;
  local control 41,232. Parameters and FLOPs are NOT matched.
- Model selection/tuning: none during this pilot.
- Metric: validation bits per byte, mean ± seed SD; lower is better.

| Model | Bits/byte (mean ± SD, n=3) |
|---|---:|
| Grassmann | 4.29101 ± 0.08849 |
| Transformer | 4.44726 ± 0.01006 |
| Non-geometric local control | **4.28297 ± 0.03246** |

This is a tiny, undertrained, 3-seed exploratory diagnostic. A model that
happens to score lower here has NOT demonstrated better language-model
quality, transfer, long-context ability or efficiency. Local mixing and
Grassmann scores are close relative to variation.

## 3. P2 bounded native Lenovo inference

Reference implementation on random weights, batch 1, FP32, rank 8,
2 layers, sequence lengths 16/32/64/128. Two warmups, five separately
synchronized repetitions per configuration. Every model's full output
was compared to its cached incremental reference within numerical
tolerance before timing. The cached path is a **Python token loop**,
not a production streaming kernel. Timings are total time to evaluate
the specified context, not milliseconds for a single generated token.

### Length 128 median timing (milliseconds)

| Device / mode | Grassmann | Transformer | Local control |
|---|---:|---:|---:|
| RTX 3050 prefill | 3.241 | **0.818** | 1.830 |
| RTX 3050 cached 128-token sweep | 436.370 | **134.244** | 222.210 |
| Lenovo CPU prefill | 2.114 | **0.688** | 1.045 |
| Lenovo CPU cached 128-token sweep | 147.292 | **58.353** | 73.756 |

Raw timings at all tested lengths 16/32/64/128, parameter counts,
cache-payload estimates and PyTorch max-allocated GPU memory are in the
JSON files. GPU profiles on random tiny models are not evidence that
attention-free architecture cannot outperform attention at other
scales with optimized kernels. Conversely, no edge speedup was observed.

## 4. Environmental safeguards and outcomes

The Python supervisor `scripts/safe_windows_runner.py` ran all trials
as managed child processes without any persistent startup automation.
Limits: temp <80°C, aggregate VRAM use <3400 MiB, free RAM >=8 GiB,
per-trial runtime <=240s; sample approximately every 2s.
It also caps each PyTorch CUDA process to 60% of device GPU memory.

| Run | Highest sampled GPU temp | Highest sampled VRAM used | Trial outcome |
|---|---:|---:|---|
| Inspect/41 tests | 36°C | 0 MiB | Passed |
| P1 synthetic | 41°C | 117 MiB | 2/2 complete |
| P1 real text | 39°C | 117 MiB | 1/1 complete |
| P2 CUDA + CPU | 43°C | 105 MiB | 2/2 complete |

Telemetry is **sampled**, so it cannot exclude an unsampled short-lived
peak. End-state after P2: 43°C, 0 MiB VRAM used, approximately 19.48 GiB
free system RAM. Battery indicated AC-powered. Observed no OOM, runaway
temperature or system instability.

Windows refused to execute the `.ps1` launcher under its default policy.
We preserved that policy and used the equally bounded native Python
supervisor instead. The Python launcher is the verified path.

## 5. Publication limitations and next scientific work

This snapshot is scientific infrastructure and exploratory empirical
evidence. Major P0/P1/P2 acceptance criteria still remain open:
faithful paper model reproduction, parameter/FLOP-matched long training
with robust multiple seeds, calibrated energy measurements, optimized
fused kernels and hardware-independent replication. The actual Xiaomi
15T has **not been accessed**, as agreed; Android work belongs to P3.

This report and raw files are public intentionally, including weaker
Grassmann timings and the data-integrity correction.

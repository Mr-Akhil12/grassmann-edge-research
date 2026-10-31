# Cross-platform CPU reproducibility record — 31 October 2026

## Source of measurements
- Repo: https://github.com/Mr-Akhil12/grassmann-edge-research
- Source revision: `267817f6c3395d0ad6577559555ecd54d88d33d8`
- CI run: https://github.com/Mr-Akhil12/grassmann-edge-research/actions/runs/41169715588
- Jobs: `ubuntu-latest` and `windows-latest`, both SUCCESS.
- Two seed trial: `11,17`; 12 training steps each model; synthetic lag=4.
- Input: 64 tokens, 4 examples, d=32, rank=8, two layers.
- Files here are unedited raw JSON artifacts from GitHub Actions.
- Data is synthetic. All models randomly initialized. Model sizes differ.
- No Lenovo RTX3050 CUDA experiments or Android tests in this record.
- No statistical comparison of model quality is justified.

## Validation cross-entropy (nats), mean of two seed runs

| Model | Ubuntu CPU | Windows CPU |
|---|---:|---:|
| Causal Grassmann | 3.63833 | 3.63833 |
| Full attention Transformer | 3.58100 | 3.58100 |
| Non-geometric local | 3.62959 | 3.62959 |

These are near-untrained short-run validation losses; ranking is not
generalizable or suitable for architecture selection.

## Untrained prefill median at sequence length 32 (milliseconds)

| Model | Ubuntu CPU | Windows CPU |
|---|---:|---:|
| Causal Grassmann | 1.0217 | 0.9281 |
| Full attention Transformer | 0.3752 | 0.2842 |
| Non-geometric local | 0.6281 | 0.6434 |

Full-sequence and cached incremental outputs agreed with maximum observed
absolute differences below 1e-6 in this CI run. Cached-loop times
include Python overhead and should not be represented as optimized decoding.

## Scientific interpretation

This public snapshot supports **cross-platform software reproducibility and
small CPU-engineering measurements only**. The current unoptimized Grassmann
implementation is slower than optimized PyTorch attention at short lengths.
This does not disprove the architecture's asymptotic benefit for longer
contexts or a fused native operator, and certainly does not prove a benefit.
Future evidence must include matched quality, actual edge device, thermal
controls, substantial training and plausible energy instrumentation.

All raw repetitions and train loss series are kept in the JSON artifacts,
including unfavorable results.

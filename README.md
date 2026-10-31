# Grassmann Edge Research

Independent research on causal Grassmann/Pluecker-based sequence mixing and whether it yields a measurable quality-versus-compute tradeoff on practical edge hardware.

**Status: open research scaffold, not a completed paper reproduction.** No edge performance advantage has been demonstrated. See [live sprint status](docs/SPRINT_STATUS.md), [P0 causality audit](docs/P0_CAUSALITY_AUDIT.md), [P0–P2 protocol](docs/SPRINT_P0_P2.md), and [preserved cross-platform raw results](experiments/2026-10-31-crossplatform-ci/README.md). The [guarded Lenovo CPU/GPU and WikiText-2 pilots](experiments/2026-10-31-lenovo-rtx3050/README.md) are also public; their limitations are described in the report.

## P2.2 endurance research

Our [locked 6.29M-parameter P2.2 protocol](docs/P22_LOCKED_PLAN.md) documents calibration, parameter-matched architecture controls, and the local finite-runtime GPU watchdog. The larger multi-seed campaign is a staged experiment; no unverified superiority claim is implied by early calibration.

## Scientific commitments

- Paper-as-written and causally corrected implementations are separate variants.
- Numerical correctness and no future-token leakage are tested before benchmarking.
- Compare Grassmann with causal Transformer and non-geometric local mixing.
- Match data, tokenizer, training budgets, and parameter size; record exceptions.
- Save all trials, including failures. Don't cherry-pick configurations or seeds.
- Measure actual inference and peak memory; report energy only with verified instrumentation.
- Keep personal machine profiles, tokens and raw hardware identifiers outside Git.

## Primary references

- Original manuscript: https://arxiv.org/abs/2512.19428
- Infatoshi reproduction: https://github.com/Infatoshi/grassmann-flows
- Kane Norman reproduction: https://github.com/kanenorman/grassmann

## Project stages

0. Verify architecture math and hardware capabilities.
1. Reproduce small, controlled experiments with robust causality checks.
2. Run hardware-limited training and bounded GPU inference on Windows.
3. Build Android CPU reference and validate operator parity.
4. Measure phone inference latency, peak memory, and available energy telemetry.
5. Publish code, results, negative findings, and reproducible research report.

## Local setup

Python dependencies must live in .venv. Use device-appropriate CUDA PyTorch from the official wheel index. See docs/first_session.md once the source archive has been copied.

Do not publish early local measurements as evidence of architectural superiority.

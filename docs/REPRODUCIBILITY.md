# Reproducibility contract

A publishable experiment must specify exact source commit, Python/PyTorch version,
seed(s), preprocessing, data split, train/val/test separation, optimizer/schedule,
architecture details, parameter count, context lengths, train tokens, duration,
device/backends, precision, measurement methodology, all raw results and errors.

Causality:
- Every model must pass prefix invariance against changes in future inputs.
- Test gradients from prefix outputs to future inputs.
- The paper's forward-pair formula is NOT assumed causal.

Fairness:
- Compare against both a causal Transformer and non-geometric local mixing.
- Equal parameter count alone is not enough; report measured FLOPs and training tokens.
- Pre-register seeds/settings; reserve held-out data for a single final evaluation.
- Publish negative and null findings.

Edge safety:
- Never change firmware, power limits, drivers, BIOS or Windows security settings.
- Do not use uncontrolled stress tests or unbounded training on a 4GB VRAM GPU.
- Read temperature, memory and process usage between bounded trials.
- Energy must be measured with calibrated telemetry or explicitly marked unavailable.

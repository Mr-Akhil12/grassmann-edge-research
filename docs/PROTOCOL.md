# Research protocol — draft, not yet preregistered

## Verified so far
Numerical unit tests, synthetic diagnostics and CUDA availability. These are
engineering checks. They do not establish model quality or edge efficiency.

## Tier 0: Correctness
- Check Pluecker antisymmetry, degenerate-pair behavior, Lagrange identity.
- Check gradients finite at layer and model scale.
- Change any input suffix and verify outputs before its start remain unchanged.
- Check gradients from early outputs to future inputs are exactly zero.
- Explicitly distinguish paper-literal forward pairing from causal correction.
- Keep an independent scalar implementation for double-precision cross-check.

## Tier 1: Controlled small models
- Compare full-attention Transformer, Grassmann, and non-geometric local pair mixing.
- Add finite-window attention and Mamba only when reference implementations
  and compute budgets are available.
- Lock tokenization, data splits, sample order, seeds, train tokens, optimizer,
  learning-rate search budget and context length across candidates.
- Match model parameters within prespecified tolerance; disclose mismatch.
- Report peak memory, elapsed GPU-hours, training throughput as well as quality.
- Explore synthetics then one small publicly redistributable language dataset.
- Confirmatory tests: at least five seeds when capacity permits, all reported.
- Hold out test until architecture/model selection is frozen.

## Tier 2: Inference mechanics
- Separate prompt prefill and autoregressive incremental decode with cache/state.
- Current full-forward script is NOT incremental decode performance evidence.
- Include context length and effective receptive field for each architecture.
- Evaluate runtime parity and floating-point numerical differences.
- Compare quality-versus-memory/latency Pareto frontiers.

## Tier 3: Android edge hardware
- Xiaomi 15T is target; device and installed OS must be verified.
- Only connect after owner's express USB debugging approval.
- Begin with CPU ARM64 reference, optimize NEON or Vulkan after parity tests.
- Do not claim NPU accelerated execution without measured operator offload.
- Randomize test order, distinguish hot/cold, repeat under controlled thermal
  conditions, collect median, p95 and confidence intervals.
- Battery % and charging input watts alone are NOT joules/token.
- Measure calibrated energy when hardware permits or mark unavailable.

## Threats to validity
The original research may contain causal-index ambiguity. A local mixer with
short receptive field is not a long-context substitute. Different optimized
kernels can swamp operation-level theoretical complexity, especially on phones.
Matching parameter count without training budget and quality is insufficient.
Small datasets, one seed or untrained models cannot justify broad conclusions.

## Release obligation
Provide command-line scripts, exact dependency versions, data checksums,
checkpoints, raw per-seed results, negative findings, hardware setup, tested
measurement methodology and honest limitations.

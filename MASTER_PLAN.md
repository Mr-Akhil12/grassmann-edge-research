# Grassmann Edge Research — Master Plan v0.2

## Mission
Determine whether causal geometric Grassmann/Pluecker sequence mixing offers
a reproducible improvement in quality versus inference cost on edge devices.
Our output is a transparent scientific result, positive, null or negative.
No integration with any existing application is planned.

## Research questions
RQ1. Are the paper's mathematical operations correctly specified and causal?
RQ2. Does a geometric mixer improve quality at matched parameters and training tokens
relative to Transformer and non-geometric local-mixing controls?
RQ3. On actual edge hardware, is there a statistically meaningful improvement in
peak memory, latency, throughput or correctly measured energy per output token?
RQ4. Which components (rank, offsets, gate, normalization, residual, kernels)
cause any observed effects?

## Stage 0: Foundation (current)
- Review paper and reproductions. Reproduce source invariants.
- Define paper-literal vs causal-corrected branches; never conflate them.
- Record stable code and package versions, scripts and run hashes.
- Run CPU tests, gradient checks, prefix invariance checks.
- Make Git repo with private machine inventory outside the tree.

## Stage 1: Small scientific models
- Same tokenizer, data splits, optimizer/schedule, training tokens and context.
- Budget model sizes as closely as possible; report deviations.
- Set seeds BEFORE runs, not after result selection.
- Train Transformer, Grassmann, non-geometric local mixer.
- Add Mamba / other efficient baselines only with comparable implementation.
- Test synthetic long-distance dependencies and WikiText-2 or other openly
  accessible language-model dataset with verifiable licenses/splits.
- Lock val/test protocol and record failures.
- Use meaningful multiple-seed experiment once implementation matures.

## Stage 2: Native GPU profiling
- Warmups, repeat counts and variance recorded.
- Separate batch prefill from incremental cached decode.
- Report parameter count, FLOPs and receptive field.
- CUDA memory guard and bounded duration; check temps and VRAM.
- Never claim random-init forward timings prove architecture superiority.

## Stage 3: Xiaomi 15T (not started)
- Obtain explicit USB debugging approval before ADB probing.
- Implement operator-equivalent CPU reference, then optimized backend.
- Run parity tests, cold/warm and thermally controlled repeats.
- Capture peak memory, prefill, decode, throttle conditions.
- Report energy only if hardware supports credible measurements; otherwise N/A.

## Stage 4: Ablations, publication
- Rank, window schedule, normalization, gate, residual, operator precision,
  kernel layouts, context length, approximate memory and energy tradeoffs.
- Publish full source, checksums, raw run tables, seeds, failures, trained
  checkpoint hashes, requirements, methodology, analysis, limitations.
- Offer scripts that reproduce figures and reports, not just screenshots.
- Consider preprint / author communication only after verification.

## Stop conditions
Immediately stop tests if GPU OOM persists, GPU temperature exceeds 80 C,
free RAM drops below 8 GiB, unexpected system instability occurs, or the
experiment fails prerequisite correctness tests. Do not alter power limits,
GPU clocks, antivirus, BIOS, firmware or installed drivers.

## Current evidence level
Research prototype only. The local source is not a faithful recreation of
the original paper's language model. No hardware benefit is established.

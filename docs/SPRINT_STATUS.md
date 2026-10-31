# P0–P2 research sprint — evidence and remaining gates (31 October 2026)

## Current status

The Lenovo is responsive and the bounded hardware experiments have now
completed. No Xiaomi 15T, ADB, Android, firmware, GPU drivers, power limits
or Windows security settings were modified or accessed during this sprint.

**Executed engineering and exploratory research is documented publicly.**
This does NOT establish superiority of Grassmann flows nor a faithful
reproduction of the original paper's 13–18 million-parameter results.

## Verified milestones

- Original paper-literal future pairing and past-only causal variant are
  implemented separately, with double-precision geometry oracle and
  future-token leakage/gradient checks.
- Incremental cached forward parity tests cover all three variants of
  interest: causal Grassmann, causal Transformer, and causal local mixing.
  Paper-literal future pairing is correctly refused for causal decode.
- Full correctness suite passed locally on the Lenovo (41 tests).
- GitHub Actions passed on clean Ubuntu/Windows CPU runners with archived
  raw numerical and timing artifacts.
- Lenovo GPU three-seed synthetic lag-4 and lag-24 experiments both completed
  under guard control without any failures.
- WikiText-2 raw train and validation files downloaded and verified against
  SHA256 hashes from the pinned immutable upstream dataset revision.
- A hash mismatch in the initial public script was caught and corrected:
  initial validation/test hashes belonged to another dataset revision.
  Neither the test split nor any test labels were used in model fitting.
- Exploratory three-seed real-text byte-level training completed on the
  Lenovo GPU with identical samples/training budget across model types.
- P2 prefill and incremental decoding references profiled on the actual
  Lenovo CPU and RTX 3050 at contexts 16,32,64,128, with raw repeats and
  GPU allocation estimates saved.
- A Python-only resource supervisor sampled GPU temperature, VRAM and free
  RAM every ~2 seconds and enforced per-child maximum runtime. Original
  Windows PowerShell execution policy remains unchanged.

## Results

Full files are under
`experiments/2026-10-31-lenovo-rtx3050/`
and `experiments/2026-10-31-crossplatform-ci/`.

### Real-text pilot (lower is better)
Mean validation bits/byte, n=3 seeds, 60 steps/model:
- Grassmann: **4.2910** (SD 0.0885)
- Transformer: **4.4473** (SD 0.0101)
- Simpler local-mixing control: **4.2830** (SD 0.0325)

Models have 41,232–43,904 parameters, are not FLOP- or parameter-matched,
and have trained on far too few tokens for architecture superiority claims.
Our 256-byte vocabulary is NOT the paper's WordPiece vocabulary. Do not
compare these numbers to the paper's WikiText-2 perplexity.

### CUDA untrained inference at context 128
Full-prefill medians: Grassmann 3.241ms, Transformer 0.818ms,
local mixer 1.830ms.
Token-loop cached whole-context-sweep medians: Grassmann 436.370ms,
Transformer 134.244ms, local mixer 222.210ms.
These are random-init reference-implementation timings, not production
hardware kernel comparisons or valid single-token generation latency.

### Machine health
Lenovo: 32GB RAM, RTX3050 Laptop 4GB, i5-11320H, driver 595.79.
Sampled max GPU temperature 43 C in P2, up to 117MiB sampled VRAM
usage in P1, no observed safety-limit triggers or OOM. Supervisor reports
0MiB GPU use after completion. Sampling does not guarantee peak capture.

## Open scientific acceptance gates (do not close issues prematurely)

**P0:** Exact paper-scale reproduction, author intent behind paper-literal
future indexing, unambiguous operation matching and larger model validation
remain open. Mathematical contradiction and corrected variant tested.

**P1:** Reliable model-quality inference needs a substantially larger,
matched-parameter and matched-FLOP/compute study, more training tokens, more
evaluation seeds, rigorous data controls, ablations and ideally independent
third-party replication. The three-seed results are *exploratory only*.

**P2:** We have actual measured Lenovo CPU/GPU reference implementation
results and telemetry. A credible scientific claim about optimized attention
vs Grassmann requires equally optimized fused/native kernels, quality-matched
trained checkpoints, a broader length regime, correct memory traffic
accounting and sound instrumentation for energy. No joules/token measured.

These unresolved gates should be described as such in all subsequent
releases. Public raw data and failure history will not be hidden.

## Next milestone: P3

The phone is **not connected or accessed**. P3 specifically needs the
user's Xiaomi 15T and explicit authorization for Android/ADB interaction.
We will not begin it without the user's involvement.

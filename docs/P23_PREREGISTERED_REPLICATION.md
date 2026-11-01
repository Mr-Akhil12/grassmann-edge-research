# P2.3 — locked cross-seed replication proposal (2026-11-01)

**Status: PREPARED ONLY. No P2.3 GPU training has started or been authorized to run unattended.** The Lenovo was offline when this protocol was drafted. Freeze the protocol and check the live machine before any trial. All previous P2.2 and thermal-abort evidence must be preserved.

## Prior evidence and immediate reproducibility gap

The exploratory P2.2c fresh 100-step seed-11 validation results (bits per byte, lower is better) were: Grassmann 3.5896828567; Transformer 3.6394865816; non-geometric Local Mixer 3.541469. The original Grassmann trial began at 37°C, and the subsequent trials used a **post-hoc** thermal-gate amendment. These results cannot be presented as a temperature-matched, multi-seed confirmation.

At this protocol's creation GitHub main ends at commit 1193fde. The locally verified v4 controller, amendment and commit ede7134, along with the local numeric results, are **not on remote main**. The missing history is a blocker to publication and P2.3 GPU execution. Sync and review the local branch/commit and selectively publish non-sensitive, audited numeric records first; never commit private hardware details or large checkpoints.

## Hypothesis, preregistered roster and estimands

Primary: for fresh 100-step runs, paired seed difference **Grassmann validation BPB minus Local Mixer BPB**; negative favors Grassmann. Secondary: Grassmann minus Transformer; report wall-clock time and actual allocated/reserved GPU memory, without claiming equal FLOPs or equal inference performance.

Exactly five predeclared seeds: **11, 17, 23, 31, 47**. The prior seed-11 runs are historical only. Rerun seed 11 fresh under the same P2.3 protocol as the other seeds; do not reuse existing checkpoints. Architecture order balances trial position:

| Seed | Frozen order |
|---:|---|
| 11 | Grassmann, Transformer, Local |
| 17 | Transformer, Local, Grassmann |
| 23 | Local, Grassmann, Transformer |
| 31 | Grassmann, Local, Transformer |
| 47 | Transformer, Grassmann, Local |

No choosing or dropping seeds based on preliminary results. Every attempt and failure must remain in the raw record.

**Gate A: systems pilot** — exactly the seed-17 triplet above, each fresh 100 optimizer steps, with a **hard 90-minute total-session ceiling**, including all cooldown time. Stop on timeout and preserve incomplete outputs. Inspect before starting another session.

**Gate B: separate later approval** — remaining 12 trials, partitioned into finite reviewed sessions. Do not automatically progress from Gate A to B, and do not authorize multi-hour unattended operation merely through this protocol.

## Frozen training configuration

- Models: causal Grassmann, causal Transformer, causal non-geometric Local Mixer; about 6.29 million parameters each (6,290,604 / 6,289,920 / 6,291,570).
- 6 layers, dimension 288, rank 28, context 512, batch 8, causal offsets 1/2/4/8/16/32, FP32.
- Per-model FFN widths 1238 / 1152 / 1399, as in P2.2. These match parameters, **not** actual compute cost or receptive field (Grassmann/local maximum ≈193 tokens vs Transformer 512).
- AdamW lr 0.0003, weight decay 0.01, gradient clip 1; exactly 100 optimizer steps from step zero, checkpoint every 20 steps, fixed seed-controlled window generator, 409,600 input-byte positions processed per trial.
- WikiText-2 raw byte stream, revision 6231e49f19a707241d6b84d9cff60a3a86b85a85; train SHA-256 c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8; validation SHA-256 714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307. Keep test split untouched.
- Eight deterministic validation windows per seed, same across model kinds; do not tune hyperparameters against this campaign.

## Gate 0: required live hardware/code review

Reconnect the Lenovo and verify actual GPU temperature and process contention, CPU/RAM, disk, AC, GPU driver, CUDA/PyTorch, Git source revision, complete checkout and original checkpoint hashes; never infer current state from yesterday's logs. Re-run math/causality/future-token leakage tests and inspect failure handling in the proposed finite supervisor. Verify original P2.2c raw manifest records and source revision. Preserve historical STOP_REQUESTED, ABORT_NOW and failed-controller evidence. No OS startup service, Windows Task Scheduler, driver, firmware, fan or power-limit changes.

## Locked thermal/safety controls until a new reviewed protocol supersedes them

- Passive starting GPU ≤33°C, GPU utilization ≤10%, VRAM ≤128 MiB, AC connected and free physical RAM ≥8 GiB.
- Active 60-second idle verification sampling around every two seconds: all observed GPU temperatures ≤37°C, utilization ≤10%, VRAM ≤128 MiB. Distinguish and preserve **passive temperature**, full active measurements and **immediate model-start temperature**. This does not claim 60 seconds at ≤33°C.
- ≥900 seconds mandatory rest between model training blocks. While waiting, GPU telemetry no more than once per 20 minutes, but owner ABORT_NOW checked about every 10 seconds without NVIDIA sampling. One training child at a time.
- During training, local supervisor checks GPU temperature/VRAM, free RAM, AC and telemetry approximately every two seconds. **Abort GPU ≥78°C, VRAM ≥3000 MiB, RAM <8 GiB, AC loss, sensor loss, non-finite loss, unexpected child exit, or >150-second child duration.** CUDA memory allocator cap ≤60%. Session cap 90 minutes for pilot.
- No automatic retry after child failure. All outputs in a fresh ignored P2.3 directory. Ambient temperature and CPU/heatsink equilibrium are not instrumented, so declare them confounders.

## Read-only analysis, uncertainty and claim limits

Use the P2.3 auditor in scripts/p23_analyze.py for complete metrics and independent checkpoint SHA-256 verification. Require identical data hashes/configuration and fresh starts; failed or missing runs cannot be quietly excluded. Report all per-seed BPB values, primary/secondary paired differences, descriptive paired bootstrap intervals, exact sign-flip sensitivity, training timing and memory. Five pairs do **not** establish universal superiority; even unanimous differences have a minimum achievable two-sided exact sign-flip p-value of 0.0625. Do not report a single pilot as statistical confirmation.

Distinguish parameter-matching from FLOPs-matching; do not infer power efficiency from uncalibrated watts, or algorithmic speed from one implementation's training timing. Any interpretation is strictly implementation-, workload-, context- and hardware-specific. Further 1000-step studies, phone experiments and publication claims require their own reviewed gates.

## Stop/resume

This file is planning, not a launcher. Do not run any GPU workload until the Lenovo is online, v4 history is synchronized, correctness and thermal preflights pass, the experiment controller is independently reviewed, and an explicit finite pilot launch is authorized.
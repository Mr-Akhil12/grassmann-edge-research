# P2.2c — Controlled thermal-reset three-model comparison

**Protocol locked: 2026-10-31, before running trials.**

## Why

Prior P2.2 runs began at successively higher GPU temperatures, creating
heat carry-over and a threshold crossing at 78°C. A separate single
cold-start retry recovered a 40->100-step checkpoint at <=57°C, but
was not an apples-to-apples architecture comparison.

This new design tests **fresh training from step 0 on all three models**
with an explicit enforced **15-minute minimum idle rest after each
model**, not just waiting until GPU temperature briefly falls.

## Study design

One owner-authorized finite, 3-model comparison on the Lenovo:
- Fixed order: causal Grassmann, causal Transformer, local-mixer control
- Seed 11 for all; each fresh 100 steps, no checkpoint reuse from P2.2
- 6 layers, width 288, rank 28, context 512, batch 8, FP32
- Parameter-matched FFN widths, mismatch <0.03%; FLOPs not matched
- WikiText-2 raw byte-level next-token task; SHA256-verified train/val
- Exactly equal training examples, random seed, optimizer, step count
- Validate only on existing validation split, test split untouched
- Outputs under ignored `results/p22/thermalreset_v1/`, no overwriting
  previous runs and no automatic retry after failure

**Thermal reset gate**, before each trial:
1. Initial cold baseline or at least **900 seconds** since the previous
   training child finished (whichever is applicable).
2. GPU <=38°C, GPU utilization <=10%, NVIDIA used VRAM <=128 MiB
   continuously for >=60 seconds.
3. AC power connected, free RAM >=8GiB and all sensors working.
4. If the reset gate fails to complete in 25 minutes, stop without
   starting the next trial. Waiting is not enough if thermal sensors
   still fail.

**During each trial**:
- Watch GPU temperature, memory, utilization, clock, NVIDIA-reported
  GPU power, available RAM and AC every ~2 seconds.
- Stop if GPU temperature >=78°C; NVIDIA used VRAM >=3000MiB;
  RAM available <8GiB; missing AC/telemetry; owner ABORT_NOW marker;
  child duration >150 seconds; or total session >55 minutes.
- Only one child at a time; kill only that child on guard failure.
- All relevant telemetry periodically flushed to local JSON with
  atomic replacement. No external scheduler, power-limit edits or
  firmware/driver/security changes.
- Historical `results/p22/STOP_REQUESTED` remains present. This
  new controller authorizes *only the specified limited trio*.

## Expected limitations

The 15-minute gap is a predeclared heat-soak mitigation, **not a
measurement of CPU package temperature or heatsink equilibrium**.
No CPU package telemetry was available in the prior work, and we have
not established the safe sustained steady-state temperature. Single
seed, 100 steps per model is engineering calibration, **not conclusive
model-quality research**. A larger multiseed campaign would need an
updated protocol and further authorization.

## Usage

Default prints the finite study design and launches nothing:

`python -m scripts.p22_thermalreset`

To execute exactly this protocol from the project virtualenv:

`python -m scripts.p22_thermalreset --execute`

To request an emergency stop, create an `ABORT_NOW` file inside
`results/p22/`; the local controller checks it during both cooldown
and training. Existing experiments and original safety markers remain.

No Xiaomi/Android action is included. P3 requires the owner's phone.

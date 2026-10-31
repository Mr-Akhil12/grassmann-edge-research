# P2.2b — cooled-start one-off GPU retest (31 October 2026)

**Outcome: passed. Full endurance study remains paused.** The original
78°C cutoff was NOT raised, the local STOP_REQUESTED marker was NOT
removed, and no multihour training was launched.

## Experiment and outcomes

After owner confirmation of elevated laptop and adequate airflow, the
Lenovo was allowed to cool to **32°C GPU core**. A narrowly scoped
guard override permitted ONLY this retrial of Grassmann seed 23 from
its 40-step checkpoint to 100 total training steps. The guard required
an initial temperature <=38°C, continued using its 78°C hard stop,
NVIDIA VRAM <3000MiB, free RAM >=8GiB, verified AC power, telemetry
every ~2 seconds, and an overall 110s wall-clock cap. A separate
ABORT_NOW marker can cancel even the one-off retest.

The original failure evidence remains untouched:
[previous 2026-10-31-P2.2 report](../2026-10-31-p22-scaled/README.md).

| Metric | Previous warm start | Cold-start retry |
|---|---:|---:|
| Starting GPU temp | 64°C | 32°C |
| Peak/stop temp | Watchdog triggered at 78°C | Max logged 57°C |
| Duration | ~28s | 35.25s |
| Model/seed | Grassmann / 23 | Grassmann / 23 |
| Work performed | Step 0 toward 100, interrupted | Resumed step 40 -> 100 |
| Outcome | Stopped automatically | Completed successfully |

During the retry, GPU utilization was 99–100% on active samples, used
VRAM peaked at 2,215MiB, loaded clocks were approximately
1,950–1,972MHz, and model validation after 100 steps was
**3.590425 bits/byte** on WikiText-2 raw byte-level validation.
No non-finite loss was reported. The original step-40 checkpoint was
copied to ignored local storage and verified SHA256:
`BC0F67B1D1CFD21F972ED7D8A03E73F205ACAE5FF4029E7D92DEFBD2ABCA931F`.
New 100-step checkpoint remains local (not published).

Afterwards: GPU 44°C idle, 0MiB used VRAM, no Python training jobs,
and the original STOP_REQUESTED safety marker remains.

## Interpretation / limitations

Starting cool made a large practical difference. This is **consistent
with heat accumulation between previous back-to-back trials**. It is
not a fully controlled experiment separating initial temperature from
ambient conditions, fans or CPU load: the previous attempt was the
initial 0-100 step run, while the retry resumed at step 40. The
successful trial was only 35 seconds, **not** a measure of the GPU's
steady-state temperature over hours. We did not instrument CPU
package temperature or calibrated energy consumption.

The GPU being reported at 86–87°C in COD gaming is important user
context but **not** authorization to lift our research cutoff. Neither
Grassmann superiority nor sustained all-day operating safety was
established. Larger work remains contingent on a separate thermal
protocol and pre-registered performance/quality study.

This directory contains the full successful guard JSON,
training metric JSON, human-readable stdout and `MANIFEST.json`
with SHA-256 hashes; model/optimizer weights remain private locally.
P3 Xiaomi work has not started.

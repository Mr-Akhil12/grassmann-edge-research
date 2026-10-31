# Contributing to Grassmann Edge Research

This is open research. Contributions are welcome when they make experiments
more falsifiable, reproducible, or efficient without hiding shortcomings.

## Before submitting work

1. Review `MASTER_PLAN.md`, `docs/SPRINT_P0_P2.md` and
   `docs/REPRODUCIBILITY.md`.
2. Open an issue for significant architectural changes, preferably a
   targeted reproduction hypothesis or an identified specification gap.
3. Use a topic branch and include deterministic unit tests. In particular,
   causal models must pass prefix-invariance and cached/full parity checks.
4. Run `python -m pytest -q` and provide your precise environment and
   output. GitHub CI also runs on clean Windows and Linux hosts.
5. If presenting benchmarks, publish raw timing records and measurement
   protocol, not only a favorable aggregated result.
6. State every difference from the paper or compared baseline.
7. Keep experimental failures and unexpected negative results visible.

## Responsible disclosure

Do not commit API keys, private hardware identifiers, personal data,
third-party checkpoint weights without distribution rights, or undocumented
external code. Cite source papers, version numbers and license terms.

## Claims and attribution

A software smoke test is **not** evidence of superior model quality. An
optimized local microbenchmark is **not** end-to-end edge throughput.
A phone temperature reading is **not** joules per generated token.

We value carefully documented failed attempts, null results and reproduction
reports just as much as positive ones.

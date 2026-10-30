# Engineering smoke checks — 30 October 2026

These four files are **raw initial diagnostics, NOT research findings**.
The architectural implementation is a causally corrected Grassmann-inspired
reference, not a faithful reproduction of the original language model.

Hardware: Windows 11, Intel Core i5-11320H, 32 GB RAM,
NVIDIA RTX 3050 Laptop 4GB, driver 595.79, PyTorch 2.10.0+cu128.
No personal machine serials or private account data included.

Initial root Git revision: 4896410. Training example:
  python -m scripts.train_pilot --device cpu --model all --steps 4 --batch 2 --length 32
  python -m scripts.train_pilot --device cuda --model all --steps 2 --batch 2 --length 32

Forward-only diagnostics:
  python -m scripts.bench_forward --device cpu --lengths 16,32,64
  python -m scripts.bench_forward --device cuda --lengths 16,32

Limitations:
- No matched model sizes or training tokens for meaningful comparison.
- Only one seed and 2-4 optimization steps; predictive losses not converged.
- Transformer and local mixer have different implementation overhead.
- No optimized CUDA kernel or cache/state for Grassmann.
- A full-forward pass is not autoregressive incremental decode.
- GPU clock state, wall power, peak GPU memory and energy were not captured.
- No statistical inference or architectural-performance conclusion is justified.

These files are retained to prevent selective omission of early negative results.

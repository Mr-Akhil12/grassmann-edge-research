# Environment and safe-running instructions

Local Windows development has user-scoped Python 3.11 and a project .venv.
Git is available via C:/Program Files/Git/cmd/git.exe.

From Windows PowerShell:
  Set-Location <the checked-out grassmann-edge-research directory>
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m scripts.train_pilot --device cpu --steps 3
  .\.venv\Scripts\python.exe -m scripts.train_pilot --device cuda --steps 2
  .\.venv\Scripts\python.exe scripts\host_probe.py

Warnings:
- Runs are tiny synthetic numerical checks, not paper experiments.
- CUDA torch wheel selected from official PyTorch cu128 index.
- Benchmarks use bounded arguments, per-process CUDA memory fraction 60%.
- No phone access or background repeat tasks have been configured.
- Hardware inventory is maintained outside the Git repository.

Confirm package integrity and actual PyTorch/CUDA state before any GPU test.

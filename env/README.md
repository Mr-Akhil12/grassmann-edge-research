# Reproducing the Windows CUDA environment

Recorded on 2026-10-30: Python 3.11.9 x64, PyTorch 2.10.0+cu128,
NumPy 2.3.5, pytest 9.2.0, RTX 3050 4GB, NVIDIA 595.79.

On a Windows machine with a compatible NVIDIA driver:

1. Create a fresh venv with Python 3.11.
2. Install CUDA wheel from the official PyTorch index:
   python -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
3. Install remaining requirements:
   python -m pip install -r env/requirements-win311-cu128.txt --extra-index-url https://download.pytorch.org/whl/cu128
4. Install the local project and tests:
   python -m pip install -e ".[dev]"
5. Run python -m pytest -q and the bounded pilot.

Versions in requirements-win311-cu128.txt were collected from the Lenovo
environment. These are Windows CUDA requirements, not cross-platform locks.
Publish per-platform lockfiles once cross-platform experiments exist.

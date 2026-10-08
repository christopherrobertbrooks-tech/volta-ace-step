# Environment

## Host

- `ember-gateway`, Ubuntu 24.04, i7-13700KF (24 threads), 16 GB RAM (15.4 GiB usable)
- Driver 580.173.02 (the last branch for Volta), CUDA toolkit 12.9 installed

## GPUs

| Index | Card | Compute | VRAM | Role |
| ---: | :--- | :--- | ---: | :--- |
| 0 | NVIDIA GeForce RTX 4070 | 8.9 (Ada) | 12 GB | Reference card (has BF16 hardware); also drives the display (~0.9 GB) |
| 1 | Tesla V100-PCIE-32GB | 7.0 (Volta) | 32 GB | The subject |

## ACE-Step

- Checkout: `ace-step/ACE-Step-1.5` at `ca1e85f` (to be updated as the work goes)
- Installed with the official `uv sync` (pins torch 2.10.0+cu128 on Linux x86_64, a prebuilt flash-attn wheel)

## Second machine (Pascal check, test 5)

- `dev-console`, Ubuntu 24.04, i5-7400 (4 threads), 24 GB RAM
- NVIDIA GeForce GTX 1070, compute 6.1 (Pascal), 8 GB -- also drives the desktop, and ~1.3 GB of it is held by a
  pinned local ollama model that was left alone, so ACE-Step had ~6 GB
- Driver 580.173.02
- ACE-Step installed the official way (`uv sync`, torch 2.10.0+cu128). **That torch has no Pascal kernels** (arch
  list starts at sm_70): "CUDA error: no kernel image is available for execution on the device". Fixed by
  reinstalling the same version from the cu126 index (`torch==2.10.0+cu126`, arch list sm_50 ... sm_90), which
  needs `--reinstall-package torch` -- a plain `uv pip install torch==2.10.0 --index-url .../cu126` is a silent no-op
  because the version number already matches.

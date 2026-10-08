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

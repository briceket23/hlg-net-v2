# Environment pinning (protocol Step 1)

Before gate G3, pin the environment of the GPU machine and commit the result:

```bash
python -m pip freeze > env/freeze.txt
nvidia-smi --query-gpu=name,driver_version --format=csv > env/gpu.txt
```

`requirements/*.txt` give the minimum versions per package; `env/freeze.txt` is the exact set used for
the confirmatory runs. The timm version and the SHA-256 of the pretrained weights actually loaded are
also written into every run's `meta.json` (`timm`, `pretrained_sha256`).

Measured on the pilot machine (13 Oct 2026): torch 2.6.0+cu124, NVIDIA GeForce RTX 4050 Laptop GPU,
Python 3.10.11, Windows 10.

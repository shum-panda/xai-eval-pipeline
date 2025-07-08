# XAI Pipeline

Project description goes here.

---

## Installation

### Important note about installing torch with CUDA 11.8

The torch package version 2.7.1+cu118 and torchvision are **not installed directly via Poetry**. This is because the official PyTorch CUDA package index (`https://download.pytorch.org/whl/cu118`) is **not a full Python package repository**. Poetry (v2.1.3) does not support selective package resolution from partial package indexes like this.

### Step-by-step installation

1. Install the regular dependencies:
   ```bash
   poetry install
   ```

2. Manually install `torch` and `torchvision` using `pip` from the official PyTorch CUDA index:
   ```bash
   poetry run pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

**Tip:** If you use a different CUDA version or platform, check the official PyTorch Get Started page for the correct URLs and versions.

### Why not add these packages directly in `pyproject.toml`?

Adding `torch` with `source = "pytorch-cu118"` in `pyproject.toml` causes Poetry to try fetching **all packages** from that source — which fails because many packages like `black` or `numpy` are not available there.

## Development

### Optional: Automate installation with a Makefile

Create a file named `Makefile` in the project root with the following content (make sure to use **tabs**, not spaces, before the commands):

```makefile
install:
	poetry install
	poetry run pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Then you can run:
```bash
make install
```

If you are using Windows or do not have `make` installed, you can run the commands manually or create a PowerShell or batch script to automate this.

## License

MIT License
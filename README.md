# XAI Evaluation Pipeline

A modular pipeline to evaluate and compare explainable AI (XAI) methods on various models and datasets.

---

## Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Dataset Setup](#dataset-setup)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Development & Contribution](#development--contribution)  
- [Testing](#testing)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

This project provides a flexible and extensible framework to evaluate, benchmark, and visualize explainability methods for machine learning models.  
It targets scientific researchers who want to systematically compare XAI techniques.

Features include:

- Support for multiple datasets and model backbones  
- Modular explainer integration (e.g., Grad-CAM, Score-CAM, Guided Backprop, etc.)  
- Evaluation metrics to quantify explanation quality  
- Visualization tools for qualitative comparison  

---

## Installation

### Prerequisites

- Python 3.10+  
- [Poetry](https://python-poetry.org/) for dependency management  

### Install the pipeline

```bash
   git clone https://github.com/shum-panda/xai-eval-pipeline.git
   cd xai-eval-pipeline
   poetry install
```

---

### 🔒 ImageNet Validation Dataset (ILSVRC2012) & Ground Truth

Due to licensing restrictions, the ImageNet dataset and its annotations **cannot be automatically downloaded** or distributed with this repository.

---

#### Setup Instructions

1. Register at [image-net.org](https://image-net.org/download-images)  
2. Download the following files from the official source:  
   - `ILSVRC2012_img_val.tar` (validation images)  
   - `ILSVRC2012_bbox_val_v3.tgz` (bounding box annotations)  
   -  `ILSVRC2012_validation_ground_truth.txt`  
3. Place all files in the folder `daten/raw/`  
4. Run the extraction script **once** to extract both images and bounding boxes:
```bash
  poetry run python src/pipeline_moduls/data/data_scripts/img_extraction_script.py
```

Resulting directory structure:
```
daten/
├── raw/
│   ├── ILSVRC2012_img_val.tar
│   ├── ILSVRC2012_bbox_val_v3.tgz
│   └── ILSVRC2012_validation_ground_truth.txt
└── extracted/
    ├── imagenet_val/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    └── bounding_boxes/
        ├── ILSVRC2012_val_00000001.xml
        └── ...
```

⚠️ The pipeline expects images in `daten/extracted/imagenet_val`, bounding box XMLs in 
`daten/extracted/bounding_boxes`, and validation labels from configured 
paths.

Running the extraction script multiple times is unnecessary unless you replace or  
add new raw archives.

## Configuration

Every experiment is one YAML file in `config/experiments/`. The YAML is the single
source of truth: everything that affects a run is set there and logged to MLflow.

Each file starts with the schema reference

```yaml
defaults:
  - base_config   # schema: src/pipe/control/utils/config_dataclasses
  - _self_
```

so Hydra validates the file against the config dataclasses before a run starts:
unknown keys (typos), wrong types and options the pipeline does not support yet
fail with a clear message instead of being ignored. See `CONFIG_README.md` for all
options.

Run an experiment:

```bash
poetry run python main.py --config-name config_resnet50_grad_cam
```

Override single values on the command line with Hydra syntax:

```bash
poetry run python main.py --config-name config_resnet50_grad_cam data.max_batches=5
```

### Reproducibility

For every run the pipeline logs to MLflow:

- the fully resolved config (`config/resolved_config.yaml` artifact)
- git commit, branch and whether the working tree had uncommitted changes
- Python, PyTorch, torchvision and CUDA versions, platform and device

Seeds for Python, NumPy and PyTorch come from `experiment.seed`;
`experiment.deterministic: true` additionally enforces deterministic PyTorch
algorithms (slower).

---

## Usage

### Single run

```bash
poetry run python main.py --config-name config_test_quick
```

### Several runs

`mulitrun_main.py` runs a list of configs one after another and creates the
comparison plots afterwards.

### Pipeline UI (node editor)

A browser UI to build experiments visually, similar to n8n: drag building blocks
onto a canvas and connect them from left to right

```
data -> model(s) -> XAI method(s) -> metric(s) -> experiment
```

Every connected model/method pair becomes one run, so several models on one
method (or several methods on one model) give the comparisons the pipeline is
built for. The building blocks are read from the registries, so a newly
registered model, XAI method or metric appears in the UI automatically.

The UI only edits configs: “Check” validates the graph against the same schema
as a hand-written YAML, “YAML” shows the generated files, and “Run” writes them to
`config/generated/<workflow>/` and starts `main.py` with each of them. Every run
can therefore be repeated without the UI. Workflows are saved as JSON in
`config/workflows/`.

```bash
# once: build the frontend (Node.js 20+)
cd ui && npm install && npm run build && cd ..

# start backend and UI, then open http://127.0.0.1:8000
poetry run python -m src.ui.server
```

For frontend development run `npm run dev` in `ui/` (Vite on port 5173, proxies
`/api` to the backend).

---

## Development & Contribution

### Development setup

```bash
   git clone https://github.com/shum-panda/xai-eval-pipeline.git
   cd xai-eval-pipeline
   poetry install
```

You can extend:

- New XAI explainers via the plugin system  
- Evaluation metrics via the metric registry  
- Dataset readers or pre-processing pipelines  

### Code style & tools

- Format code with [Black](https://github.com/psf/black)  
- Lint code using [Ruff](https://github.com/charliermarsh/ruff)  
- Type checking is done with `mypy` (if configured)

---

## Testing

```bash
poetry run pytest
```

- `tests/unit/`: fast tests without ImageNet or GPU (metrics, data loading,
  all XAI methods on a tiny CNN, config schema, provenance, UI backend)
- `tests/integration/`: evaluator on whole batches
- `tests/archive/`: outdated tests for old APIs, not collected

The CI runs ruff, pytest and mypy on every push.

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, ideas, or contributions:

- **Shium Mohammed Rahman** – shium.m.r@gmail.com  
- GitHub: [shum-panda](https://github.com/shum-panda)

---
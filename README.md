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

Pipeline behavior is controlled via YAML config files located in `configs/`.

You can configure:

- Dataset paths and parameters  
- Model architectures and checkpoints  
- Explainer types and hyperparameters  
- Evaluation metrics  
- Visualization settings  

To run the pipeline with a specific config:

```bash
  poetry run python main.py --config configs/example.yaml
```

You can override individual parameters via CLI or Hydra syntax if needed.

---

## Usage

### Full pipeline run

```bash
  poetry run python main.py
```

### Running specific modules

You can also run stages like explanation generation or evaluation independently by adjusting your config or using subcommands.

*Add usage patterns or command options as needed.*

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

The project uses `pytest` for testing. To run all tests:

```bash
  poetry run pytest tests/
```

*Test structure and coverage notes can be added here.*

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
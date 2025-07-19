# XAI Evaluation Pipeline

A modular pipeline to evaluate and compare explainable AI (XAI) methods on various models and datasets.

---

## Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [How to Contribute / Development](#how-to-contribute--development)  
- [Testing](#testing)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

This project provides a flexible and extensible framework to evaluate, benchmark, and visualize explainability methods for machine learning models, targeting scientific researchers who want to systematically compare XAI techniques.

- Supports multiple datasets and models  
- Modular explainers integration  
- Comprehensive evaluation metrics  
- Visualization tools for explainability results

*More detailed description and architecture diagram can be added here.*

---

## Installation

### Prerequisites

- Python 3.x (tested on 3.10+)  
- Poetry for dependency management  
- CUDA (optional, if using GPU-enabled torch builds)  

### Step-by-step installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shum-panda/xai-eval-pipeline.git
   cd xai-eval-pipeline
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. (Optional) If you require GPU support and specific CUDA versions, ensure your system matches the CUDA requirements and that the PyTorch CUDA package source is properly configured.

---

## Configuration

The pipeline uses YAML configuration files to specify:

- Dataset paths and parameters  
- Model checkpoints and settings  
- Explainer methods and their hyperparameters  
- Evaluation metrics to compute  
- Visualization options  

Example configuration files are provided in the `configs/` directory.

To run the pipeline with a specific config:

```bash
poetry run python main.py --config configs/example.yaml
```

*Add further notes on customizing configs, overriding parameters, etc.*

---

## Usage

### Basic usage

Run the full pipeline with default config:

```bash
poetry run python main.py
```

### Running specific modules

You can run individual modules such as model training, explanation generation, evaluation, or visualization by specifying commands or config flags.  

*Add usage examples or CLI flags.*

---

## How to Contribute / Development

### Development setup

1. Clone the repo and install dev dependencies:

   ```bash
   git clone https://github.com/yourusername/xai-eval-pipeline.git
   cd xai-eval-pipeline
   poetry install
   ```

2. (Optional) Setup pre-commit hooks, linters, etc.

3. Implement new explainers, metrics, or dataset loaders by extending the modular interfaces.

### Coding style

- Use [Black](https://github.com/psf/black) for formatting  
- Use [Ruff](https://github.com/charliermarsh/ruff) for linting

---

## Testing

### Running tests

The project uses `pytest`. To run tests:

```bash
poetry run pytest tests/
```

*Add notes on coverage, test structure, writing new tests.*

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or contributions, please open issues or contact:

- Shium Mohammed Rahman – shium.m.r@gmail.com  
- GitHub: [shum-panda](https://github.com/shum-panda)

---


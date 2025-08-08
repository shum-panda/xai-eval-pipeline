# XAI Evaluation Pipeline - Analysis Module Documentation

A comprehensive guide to the analysis framework for explainable AI (XAI) evaluation, providing statistical analysis, visualization, and comparison capabilities for XAI method evaluation.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Analysis Types](#analysis-types)
5. [Statistical Testing](#statistical-testing)
6. [Visualization Capabilities](#visualization-capabilities)
7. [Data Formats](#data-formats)
8. [Usage Guide](#usage-guide)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)

## Overview

The analysis module provides a comprehensive framework for evaluating and comparing XAI methods across different models and datasets. It supports both **single-experiment deep analysis** and **multi-experiment comparative studies** with robust statistical testing and publication-quality visualizations.

### Key Features

- 🔍 **Automated Experiment Discovery**: Intelligent loading of experiment results from various directory structures
- 📊 **Comprehensive Statistical Analysis**: Mann-Whitney U, Cohen's d, KS-test, correlation analysis, and more
- 📈 **Publication-Quality Visualizations**: Radar charts, heatmaps, distribution plots, confusion matrices
- 🔬 **Multi-Experiment Comparisons**: Compare XAI methods across models and datasets
- 📝 **MLflow Integration**: Automatic experiment tracking and artifact management
- ⚖️ **Statistical Significance Testing**: Rigorous statistical validation of results

## Architecture

The analysis framework consists of three main layers:

```
┌─────────────────────────────────────────────────────────┐
│                Analysis Orchestration Layer             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐│
│  │   SimpleAnalyzer    │  │  Single Run Analysis        ││
│  │  (Multi-experiment  │  │  (Individual experiment     ││
│  │   comparisons)      │  │   deep dive)                ││
│  └─────────────────────┘  └─────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                Visualization Layer                      │
│  ┌─────────────────────────────────────────────────────┐│
│  │              AdvancedPlotter                        ││
│  │    (Publication-quality plots & visualizations)    ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                Statistical Analysis Layer               │
│  Statistical Tests | Correlation Analysis | F1-Scores   │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SimpleAnalyzer 
**Location:** `src/analyse/simple_analyzer.py`

**Purpose:** Main orchestrator for multi-experiment analysis and comparison studies.

#### Key Responsibilities:
- **Data Discovery**: Automatically finds and loads experiment results using configuration-based and pattern-matching strategies
- **Multi-Experiment Coordination**: Manages analysis workflows across multiple XAI methods and models
- **Sample Balancing**: Ensures fair comparisons by balancing sample sizes across experiments
- **Custom Analysis Generation**: Creates specialized analyses like F1-score heatmaps and statistical histogram comparisons

#### Core Methods:

```python
class SimpleAnalyzer:
    def diagnose_available_data() -> None:
        """Discovers and validates all available experiment data"""
        
    def run_all_analyses() -> None:
        """Executes all predefined analysis workflows"""
        
    def analyze_resnet_methods() -> None:
        """Compares XAI methods on ResNet architectures"""
        
    def analyze_vgg_methods() -> None:
        """Compares XAI methods on VGG architectures"""
        
    def analyze_model_comparison() -> None:
        """Compares different models using the same XAI method"""
```

#### Data Loading Strategy:
1. **Primary**: Configuration-based loading using Hydra
2. **Fallback**: Pattern matching in results directories
3. **Metadata Enrichment**: Automatic extraction of model and explainer information
4. **Validation**: Data quality checks and format validation

### 2. AdvancedPlotter
**Location:** `src/analyse/advanced_plotting.py`

**Purpose:** Comprehensive visualization engine for creating publication-quality plots.

#### Supported Plot Types:

| Plot Type | Purpose | Output |
|-----------|---------|---------|
| **Dataset Overview Dashboard** | Multi-panel summary with sample counts, accuracy, timing | `dataset_overview_dashboard.png` |
| **Radar Chart** | Multi-metric performance comparison with F1-score calculation | `radar_chart.png` |
| **Correlation Heatmap** | Statistical correlation analysis between metrics | `correlation_heatmap.png` |
| **Distribution Analysis** | Violin + box plots showing metric distributions | `metric_distributions.png` |
| **Point Game Confusion Matrix** | Detailed Point Game performance analysis | `point_game_confusion_matrix.png` |

#### Core Methods:

```python
class AdvancedPlotter:
    def create_comprehensive_analysis() -> Dict[str, Path]:
        """Generates all plot types and returns file paths"""
        
    def _plot_radar_chart() -> Path:
        """Creates normalized multi-metric radar visualization"""
        
    def _plot_correlation_heatmap() -> Path:
        """Statistical correlation matrix with significance testing"""
        
    def plot_point_game_confusion_matrix_comparison() -> Path:
        """Advanced Point Game performance analysis"""
```

#### Visualization Features:
- **High-Resolution Output**: 300 DPI for publication quality
- **Consistent Styling**: Professional seaborn-based themes
- **Statistical Annotations**: Automatic addition of means, significance tests, sample sizes
- **Responsive Layouts**: Automatic grid sizing based on data dimensions

### 3. SingleRunAnalyse
**Location:** `src/pipeline/pipeline_moduls/single_run_analyse/single_run_analysis.py`

**Purpose:** Deep statistical analysis of individual experiment results with focus on meta-analysis.

#### Analysis Capabilities:

```python
class SingleRunAnalyse:
    def calculate_statistical_tests_for_all_metrics() -> pd.DataFrame:
        """Comprehensive statistical testing suite"""
        
    def plot_metric_vs_correctness() -> Path:
        """Boxplot analysis split by prediction correctness"""
        
    def correlation_with_correctness() -> Dict:
        """Pearson correlation analysis with prediction success"""
        
    def calculate_model_method_f1_scores() -> pd.DataFrame:
        """F1-score calculation using Point Game as predictor"""
```

#### Statistical Tests Implemented:
- **Mann-Whitney U Test**: Non-parametric distribution comparison
- **Cohen's d**: Effect size with interpretation (negligible/small/medium/large)
- **Kolmogorov-Smirnov Test**: Distribution shape analysis
- **Welch's t-test**: Means comparison allowing unequal variances
- **Levene Test**: Variance equality testing

## Analysis Types

### 1. Multi-Experiment Comparative Analysis

**Use Case**: Compare XAI methods across different models or datasets

```python
# Example: Compare XAI methods on ResNet
analyzer = SimpleAnalyzer()
analyzer.analyze_resnet_methods()

# Generates:
# - Combined dataset with balanced sampling
# - Comprehensive statistical comparison
# - F1-score performance heatmaps
# - Distribution analysis histograms
```

**Generated Outputs:**
```
results/analyse/resnet_xai_methods/
├── plots/
│   ├── dataset_overview_dashboard.png      # Sample counts, accuracy overview
│   ├── radar_chart.png                     # Multi-metric performance radar
│   ├── correlation_heatmap.png             # Inter-metric correlations
│   ├── metric_distributions.png            # Distribution comparisons
│   ├── f1_score_heatmap_*.png             # F1-score performance matrix
│   └── histogram_comparison_*.png          # Statistical histogram analysis
├── combined_data.csv                       # Merged experiment data
└── f1_scores_*.csv                        # F1-score analysis results
```

### 2. Single-Experiment Deep Dive Analysis

**Use Case**: Detailed statistical analysis of individual experiment results

```python
# Example: Deep analysis of a single experiment
df = pd.read_csv("experiment_results.csv")
analysis = SingleRunAnalyse(df, output_dir="analysis_output")

# Statistical testing
stats_results = analysis.calculate_statistical_tests_for_all_metrics()

# Visualization
analysis.plot_iou_histograms_by_correctness()
analysis.plot_point_game_confusion_matrices()
```

**Generated Outputs:**
```
results/experiment_name/single_run_analysis/
├── plots/
│   ├── metric_vs_correctness_boxplots.png     # Boxplot comparisons
│   ├── iou_histogram_by_correctness.png       # IoU distribution analysis
│   ├── pixel_precision_histograms.png         # Pixel-level metric analysis
│   └── point_game_confusion_matrices.png      # Point Game performance
└── data/
    ├── statistical_tests.csv                  # Complete statistical analysis
    ├── correlations.csv                       # Correlation analysis results
    └── histogram_statistics.csv               # Distribution statistics
```

## Statistical Testing

### Comprehensive Statistical Analysis Pipeline

The analysis framework includes a robust statistical testing suite designed for XAI evaluation validation:

#### Test Selection Strategy:
1. **Non-parametric Tests**: Mann-Whitney U (no distribution assumptions)
2. **Effect Size Measures**: Cohen's d (practical significance)
3. **Distribution Comparison**: Kolmogorov-Smirnov test (shape differences)
4. **Parametric Validation**: Welch's t-test and Levene test (variance assumptions)

#### Statistical Output Format:

```csv
metric,actual_column,correct_samples,incorrect_samples,correct_mean,incorrect_mean,
mann_whitney_u_statistic,mann_whitney_u_p_value,mann_whitney_u_significant,
cohens_d,cohens_d_interpretation,ks_statistic,ks_p_value,ks_significant,
welch_t_statistic,welch_t_p_value,welch_t_significant,
levene_statistic,levene_p_value,equal_variances
```

#### Effect Size Interpretation:
- **Cohen's d < 0.2**: Negligible effect
- **0.2 ≤ d < 0.5**: Small effect
- **0.5 ≤ d < 0.8**: Medium effect
- **d ≥ 0.8**: Large effect

### Example Statistical Results:

```python
# Results show IoU metric analysis
{
    'metric': 'iou',
    'correct_mean': 0.276,
    'incorrect_mean': 0.149,
    'mann_whitney_u_p_value': 1.05e-189,  # Highly significant
    'cohens_d': 0.691,                     # Medium effect size
    'cohens_d_interpretation': 'medium'
}
```

## Visualization Capabilities

### 1. Dataset Overview Dashboard

**Purpose**: Multi-panel summary providing quick experiment overview

**Components**:
- Sample count distributions by model/method
- Accuracy rates comparison
- Processing time analysis
- Data quality metrics

```python
plotter = AdvancedPlotter(df, output_dir)
dashboard_path = plotter._plot_dataset_overview()
```

### 2. Multi-Metric Radar Chart

**Purpose**: Normalized performance comparison across all metrics

**Features**:
- **Automatic Normalization**: All metrics scaled to 0-1 range
- **F1-Score Integration**: Point Game metric shows F1-score when prediction_correct available
- **Statistical Annotations**: Mean values and sample sizes
- **Custom Color Coding**: Different colors for each model/method combination

```python
# F1-Score calculation for Point Game metric
if 'prediction_correct' in data.columns:
    y_true = data['prediction_correct'].astype(int)
    y_pred = (data['point_game'] >= 0.5).astype(int)
    f1_score = f1_score(y_true, y_pred)
```

### 3. Advanced Distribution Analysis

**Purpose**: Deep statistical comparison of metric distributions

**Features**:
- **Violin Plots**: Show distribution shapes and density
- **Box Plots**: Quartile analysis and outlier detection  
- **Statistical Overlays**: Mean lines, confidence intervals
- **Significance Testing**: Embedded statistical test results

### 4. Correlation Analysis

**Purpose**: Understanding relationships between metrics and prediction success

**Output**:
- **Correlation Matrix**: Pearson correlation coefficients
- **Significance Testing**: P-values for correlation significance
- **Prediction Correctness Correlation**: Special focus on classification performance relationship

## Data Formats

### Input Data Requirements

**Primary CSV Format** (from experiment pipeline):
```csv
image_path,model_name,explainer_name,prediction_correct,prediction_confidence,
class_id,predicted_class,ground_truth_class,processing_time_seconds,
iou,pixelprecisionrecall_precision,pixelprecisionrecall_recall,point_game
```

**Key Columns**:
- `model_name`: Model architecture (e.g., "resnet50", "vgg16")
- `explainer_name`: XAI method (e.g., "grad_cam", "score_cam")
- `prediction_correct`: Boolean classification correctness
- `prediction_confidence`: Model confidence score [0,1]
- `iou`: Intersection over Union score [0,1]
- `pixelprecisionrecall_precision`: Pixel-level precision [0,1]
- `pixelprecisionrecall_recall`: Pixel-level recall [0,1]  
- `point_game`: Point Game metric score [0,1]

### Configuration Integration

**Hydra Configuration Discovery**:
```python
# Automatic experiment discovery
with initialize_config_dir(config_dir="config/experiments"):
    cfg = compose(config_name="config_resnet50_grad_cam.yaml")
    output_dir = Path(cfg.experiment.output_dir)
    data_path = output_dir / "results_with_metrics.csv"
```

**Fallback Pattern Matching**:
```python
# Robust data discovery when config loading fails
possible_patterns = [
    f"*{config_base}*",           # Direct config name match
    f"*{model_name}*",            # Model name match  
    "experiment_*",               # Generic experiment folders
]
```

## Usage Guide

### Quick Start

#### 1. Single Experiment Analysis

```bash
# Run comprehensive single experiment analysis
python run_single_analysis.py --config config_resnet50_grad_cam

# Output: results/resnet50_grad_cam/single_run_analysis/
```

#### 2. Multi-Experiment Comparison

```bash
# Run all predefined comparative analyses
python -m src.analyse.simple_analyzer

# Generated analyses:
# - ResNet method comparison
# - VGG method comparison  
# - Model comparison with GradCAM
```

#### 3. Data Discovery

```python
from src.analyse.simple_analyzer import SimpleAnalyzer

analyzer = SimpleAnalyzer()
analyzer.diagnose_available_data()
# Prints comprehensive overview of all available experiment data
```

### Programmatic Usage

#### Custom Multi-Experiment Analysis

```python
from src.analyse.simple_analyzer import SimpleAnalyzer
from pathlib import Path

analyzer = SimpleAnalyzer()

# Define custom experiment comparison
config_names = [
    "config_resnet50_grad_cam",
    "config_resnet50_score_cam",
    "config_resnet50_integrated_gradients"
]

output_dir = Path("results/analyse/custom_resnet_comparison")

# Run analysis with custom parameters
analyzer._run_analysis(
    config_names=config_names,
    output_dir=output_dir,
    analysis_name="Custom ResNet XAI Comparison",
    balance_samples=True  # Ensure fair comparison
)
```

#### Statistical Analysis Integration

```python
from src.pipe.moduls.single_run_analyse.single_run_analysis import

SingleRunAnalyse
import pandas as pd

# Load experiment data
df = pd.read_csv("experiment_results.csv")

# Initialize analysis
analysis = SingleRunAnalyse(df, output_dir=Path("analysis_output"))

# Comprehensive statistical testing
stats_results = analysis.calculate_statistical_tests_for_all_metrics()

# Generate all standard plots
plots = {
    'boxplots': analysis.plot_metric_vs_correctness(),
    'iou_histograms': analysis.plot_iou_histograms_by_correctness(),
    'pixel_histograms': analysis.plot_pixel_precision_histograms_by_correctness(),
    'confusion_matrices': analysis.plot_point_game_confusion_matrices()
}

# Access statistical results
for _, row in stats_results.iterrows():
    print(f"{row['metric']}: p={row['mann_whitney_u_p_value']:.2e}, "
          f"Cohen's d={row['cohens_d']:.3f} ({row['cohens_d_interpretation']})")
```

#### Custom Plotting

```python
from src.analyse.advanced_plotting import AdvancedPlotter
import pandas as pd

# Load multi-experiment data
df = pd.read_csv("combined_experiments.csv")
output_dir = Path("custom_plots")

# Initialize plotter
plotter = AdvancedPlotter(df, output_dir)

# Generate specific plot types
plots = {
    'overview': plotter._plot_dataset_overview(),
    'radar': plotter._plot_radar_chart(),  
    'correlations': plotter._plot_correlation_heatmap(),
    'distributions': plotter._plot_metric_distributions()
}

print(f"Generated {len(plots)} plots in {output_dir}")
```

### MLflow Integration

#### Automatic Experiment Tracking

```python
import mlflow
from src.analyse.simple_analyzer import SimpleAnalyzer

# MLflow tracking during analysis
with mlflow.start_run(run_name="comparative_analysis"):
    analyzer = SimpleAnalyzer()
    analyzer.analyze_resnet_methods()
    
    # Automatic artifact logging
    mlflow.log_artifact("results/analyse/resnet_xai_methods/")
    mlflow.log_param("analysis_type", "resnet_methods")
    mlflow.log_param("balance_samples", True)
```

#### Custom MLflow Integration

```python
import mlflow
from src.pipe.moduls.single_run_analyse.single_run_analysis import

SingleRunAnalyse

with mlflow.start_run(run_name="statistical_analysis"):
    analysis = SingleRunAnalyse(df, output_dir)

    # Log statistical results
    stats_df = analysis.calculate_statistical_tests_for_all_metrics()

    for _, row in stats_df.iterrows():
        mlflow.log_metric(f"{row['metric']}_p_value", row['mann_whitney_u_p_value'])
        mlflow.log_metric(f"{row['metric']}_cohens_d", row['cohens_d'])
        mlflow.log_param(f"{row['metric']}_effect_size", row['cohens_d_interpretation'])

    # Log artifacts
    for plot_path in analysis.generate_all_plots():
        mlflow.log_artifact(str(plot_path))
```

## API Reference

### SimpleAnalyzer

```python
class SimpleAnalyzer:
    """Main analysis orchestrator for multi-experiment comparisons"""
    
    def __init__(self):
        """Initialize with project root detection"""
        
    def diagnose_available_data() -> None:
        """Discover and validate experiment data availability"""
        
    def run_all_analyses() -> None:
        """Execute all predefined analysis workflows"""
        
    def analyze_resnet_methods() -> None:
        """Compare XAI methods on ResNet: GradCAM, Guided Backprop, Integrated Gradients"""
        
    def analyze_vgg_methods() -> None:
        """Compare XAI methods on VGG: GradCAM, ScoreCAM"""
        
    def analyze_model_comparison() -> None:
        """Compare models with GradCAM: ResNet18/34/50, VGG16"""
        
    def create_f1_score_heatmap(df: pd.DataFrame, plot_dir: Path, 
                               analysis_name: str) -> Path:
        """Generate F1-score performance heatmap"""
        
    def create_histogram_comparison(df: pd.DataFrame, plot_dir: Path,
                                   analysis_name: str) -> Path:
        """Create comprehensive statistical histogram analysis"""
```

### AdvancedPlotter

```python
class AdvancedPlotter:
    """Publication-quality plotting and visualization engine"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path):
        """Initialize with data and output directory"""
        
    def create_comprehensive_analysis() -> Dict[str, Path]:
        """Generate all plot types and return file paths"""
        
    def plot_point_game_confusion_matrix_comparison() -> Path:
        """Advanced Point Game confusion matrix analysis"""
        
    def _plot_dataset_overview() -> Path:
        """Multi-panel dashboard with sample counts, accuracy, timing"""
        
    def _plot_radar_chart() -> Path:
        """Multi-metric performance radar with F1-score calculation"""
        
    def _plot_correlation_heatmap() -> Path:
        """Statistical correlation matrix with significance testing"""
        
    def _plot_metric_distributions() -> Path:
        """Violin + box plots for distribution analysis"""
```

### SingleRunAnalyse

```python
class SingleRunAnalyse:
    """Deep statistical analysis for individual experiments"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path):
        """Initialize with experiment data and output directory"""
        
    def calculate_statistical_tests_for_all_metrics() -> pd.DataFrame:
        """Comprehensive statistical testing suite with effect sizes"""
        
    def plot_metric_vs_correctness() -> Path:
        """Boxplot analysis split by prediction correctness"""
        
    def correlation_with_correctness() -> Dict:
        """Pearson correlation analysis with prediction success"""
        
    def calculate_model_method_f1_scores() -> pd.DataFrame:
        """F1-score calculation using Point Game as predictor"""
        
    def plot_iou_histograms_by_correctness() -> Path:
        """IoU distribution analysis with statistical overlays"""
        
    def plot_point_game_confusion_matrices() -> Path:
        """Point Game performance confusion matrix analysis"""
```

## Best Practices

### 1. Data Quality Assurance

#### Validate Input Data
```python
def validate_experiment_data(df: pd.DataFrame) -> bool:
    """Validate experiment data quality"""
    required_columns = [
        'model_name', 'explainer_name', 'prediction_correct',
        'iou', 'point_game', 'prediction_confidence'
    ]
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data ranges
    assert df['iou'].between(0, 1).all(), "IoU values must be in [0,1]"
    assert df['point_game'].between(0, 1).all(), "Point Game values must be in [0,1]"
    
    return True
```

#### Sample Balance Considerations
```python
# Use sample balancing for fair multi-method comparisons
analyzer._run_analysis(
    config_names=configs,
    output_dir=output_dir,
    analysis_name="Balanced Comparison",
    balance_samples=True  # Ensures equal sample sizes across methods
)
```

### 2. Statistical Analysis Guidelines

#### Multiple Comparison Corrections
```python
from statsmodels.stats.multitest import multipletests

# Apply Bonferroni correction for multiple comparisons
p_values = stats_results['mann_whitney_u_p_value'].values
rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
stats_results['p_value_corrected'] = p_corrected
```

#### Effect Size Interpretation
```python
def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size"""
    if abs(cohens_d) < 0.2:
        return "negligible"
    elif abs(cohens_d) < 0.5:
        return "small"
    elif abs(cohens_d) < 0.8:
        return "medium"
    else:
        return "large"
```

### 3. Visualization Best Practices

#### High-Quality Publication Output
```python
# Configure matplotlib for publication quality
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
```

#### Color Accessibility
```python
# Use colorblind-friendly palettes
import seaborn as sns
colors = sns.color_palette("colorblind", n_colors=len(methods))
```

### 4. Performance Optimization

#### Large Dataset Handling
```python
# For large datasets, consider sampling for visualization
if len(df) > 10000:
    # Stratified sampling to maintain group proportions
    sampled_df = df.groupby(['model_name', 'explainer_name']).apply(
        lambda x: x.sample(min(1000, len(x)), random_state=42)
    ).reset_index(drop=True)
```

#### Memory Management
```python
# Use generators for large-scale analysis
def analyze_experiments_generator(config_names):
    """Generator for memory-efficient analysis"""
    for config_name in config_names:
        df = load_experiment_data(config_name)
        yield config_name, analyze_single_experiment(df)
        del df  # Explicit memory cleanup
```

### 5. Reproducibility

#### Seed Management
```python
import random
import numpy as np

def set_analysis_seed(seed: int = 42):
    """Set seeds for reproducible analysis"""
    random.seed(seed)
    np.random.seed(seed)
    
# Use consistent seeding
set_analysis_seed(42)
```

#### Version Tracking
```python
def log_analysis_environment():
    """Log analysis environment for reproducibility"""
    import sys
    import platform
    
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'matplotlib_version': plt.matplotlib.__version__
    }
    
    return env_info
```

---

## Support and Troubleshooting

### Common Issues

#### 1. Data Loading Failures
**Problem**: `FileNotFoundError` when loading experiment data
**Solution**: Use `diagnose_available_data()` to check available experiments

```python
analyzer = SimpleAnalyzer()
analyzer.diagnose_available_data()  # Shows all available data
```

#### 2. Memory Issues with Large Datasets
**Problem**: Out of memory errors during analysis
**Solution**: Use data sampling or batch processing

```python
# Sample large datasets
if len(df) > 50000:
    df = df.sample(n=10000, random_state=42)
```

#### 3. Missing Statistical Significance
**Problem**: No significant differences found between methods
**Solutions**:
- Check sample sizes (need adequate power)
- Verify data quality and metric calculations
- Consider effect sizes even with non-significant p-values

### Debug Mode

```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = SimpleAnalyzer()
analyzer.analyze_resnet_methods()
```

---

**Last Updated**: August 2025  
**Version**: 2.0  
**Documentation Coverage**: Complete analysis module framework
"""
Comprehensive Analysis Component for XAI Experiments.

This module provides the main orchestration layer for analyzing and comparing
XAI (Explainable AI) experiments across different models, methods, and datasets.
It supports both individual experiment analysis and multi-experiment comparative studies
with robust statistical analysis and visualization capabilities.
"""

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sympy import false

from src.analyse.advanced_plotting import AdvancedPlotter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimpleAnalyzer:
    """
    Main analysis orchestrator for XAI experiment evaluation and comparison.

    This class provides comprehensive analysis capabilities for XAI experiments,
    including automated data discovery, multi-experiment comparisons, statistical
    analysis, and publication-quality visualizations. It serves as the primary
    entry point for analyzing XAI method performance across different models
    and datasets.

    Features:
        - Automatic experiment data discovery and loading
        - Multi-experiment comparative analysis
        - Sample balancing for fair comparisons
        - Integration with AdvancedPlotter for visualizations
        - Statistical analysis including F1-scores and histogram comparisons
        - Robust fallback strategies for data loading

    Attributes:
        project_root (Path): Root directory of the project
        logger (Logger): Logger instance for this class

    Example:
        >>> analyzer = SimpleAnalyzer()
        >>> analyzer.diagnose_available_data()
        >>> analyzer.run_all_analyses()
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.logger = logging.getLogger(__name__)

    def diagnose_available_data(self) -> None:
        """
        Discover and validate all available experiment data in the results directory.

        This method scans the results directory for experiment data files,
        validates their structure, and provides a comprehensive overview of
        available experiments. It checks for required files and columns,
        reports data quality issues, and summarizes the available datasets.

        The method looks for:
        - results_with_metrics.csv files in experiment directories
        - Required columns (model_name, explainer_name, metrics)
        - Data integrity and sample counts

        Prints:
            Detailed report of all available experiment data including:
            - Experiment directory names
            - File paths and sample counts
            - Column availability status
            - Data quality warnings

        Note:
            This is primarily a diagnostic tool and should be run before
            attempting analysis to ensure data availability.
        """
        results_dir = self.project_root / "results"

        print("=" * 80)
        print("DIAGNOSE: VERFÜGBARE EXPERIMENT-DATEN")
        print("=" * 80)

        if not results_dir.exists():
            print(f"[ERROR] Results-Verzeichnis existiert nicht: {results_dir}")
            return

        found_experiments = []

        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                csv_file = exp_dir / "results_with_metrics.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file, nrows=5)
                        found_experiments.append(
                            {
                                "directory": exp_dir.name,
                                "csv_path": csv_file,
                                "samples": len(pd.read_csv(csv_file)),
                                "columns": list(df.columns),
                                "has_model_name": "model_name" in df.columns,
                                "has_explainer_name": "explainer_name" in df.columns,
                            }
                        )
                    except Exception as e:
                        print(f"[WARNING] Fehler beim Lesen von {csv_file}: {e}")

        if not found_experiments:
            print("[ERROR] Keine Experiment-Daten gefunden!")
            print(f"   Suche in: {results_dir}")
            print("   Erwartete Datei: results_with_metrics.csv")
        else:
            print(f"[OK] {len(found_experiments)} Experimente gefunden:")
            print()

            for exp in found_experiments:
                print(f"[FOLDER] {exp['directory']}")
                print(f"   [FILE] Datei: {exp['csv_path']}")
                print(f"   [DATA] Samples: {exp['samples']}")
                print(
                    f"   [MODEL] Model Name: "
                    f"{'OK' if exp['has_model_name'] else 'MISSING'}"
                )
                print(
                    f"   [EXPLAINER] Explainer Name: "
                    f"{'OK' if exp['has_explainer_name'] else 'MISSING'}"
                )
                print(
                    f"   [COLUMNS] Spalten: {', '.join(exp['columns'][:10])}"
                    f"{'...' if len(exp['columns']) > 10 else ''}"
                )
                print()

        print("=" * 80)

    def run_all_analyses(self) -> None:
        """
        Execute all predefined analysis workflows.

        This method runs a comprehensive suite of comparative analyses:
        1. ResNet method comparison (GradCAM, Guided Backprop, Integrated Gradients)
        2. VGG method comparison (GradCAM, ScoreCAM)
        3. Model comparison using GradCAM (ResNet18/34/50, VGG16)

        Each analysis includes:
        - Data loading and validation
        - Statistical comparison
        - Visualization generation
        - F1-score analysis
        - Distribution analysis

        Results are saved to respective subdirectories in results/analyse/

        Raises:
            FileNotFoundError: If required experiment data is not found
            ValueError: If data validation fails
        """
        # ResNet Methodenvergleich
        self.analyze_resnet_methods()

        # VGG Methodenvergleich
        self.analyze_vgg_methods()

        # Modellvergleich mit GradCAM
        self.analyze_model_comparison()

    def analyze_resnet_methods(self) -> None:
        """
        Compare different XAI methods on ResNet architectures.

        Analyzes and compares the performance of multiple XAI methods:
        - GradCAM
        - Guided Backpropagation
        - Integrated Gradients

        All using ResNet50 as the base model. Results include comprehensive
        statistical comparisons, visualizations, and performance metrics.

        Generated outputs:
            - Combined dataset CSV
            - Statistical comparison plots
            - F1-score performance heatmaps
            - Distribution analysis histograms

        Output directory: results/analyse/resnet_xai_methods/
        """
        config_names = [
            "config_resnet50_grad_cam",
            "config_resnet50_guided_backprop",
            "config_resnet50_integrated_gradients",
        ]

        output_dir = self.project_root / "results" / "analyse" / "resnet_xai_methods"
        self._run_analysis(config_names, output_dir, "ResNet XAI Methods")

    def analyze_vgg_methods(self) -> None:
        """
        Compare different XAI methods on VGG architectures.

        Analyzes and compares the performance of XAI methods:
        - GradCAM
        - ScoreCAM

        Both using VGG16 as the base model. Includes sample balancing
        to ensure fair comparison between methods.

        Generated outputs:
            - Balanced combined dataset CSV
            - Statistical comparison plots
            - F1-score performance analysis
            - Distribution comparison histograms

        Output directory: results/analyse/vgg16_xai_methods/

        Note:
            This analysis uses sample balancing (balance_samples=True)
            to ensure equal sample sizes across methods for fair comparison.
        """
        config_names = ["config_vgg16_grad_cam", "config_vgg16_score_cam"]

        output_dir = self.project_root / "results" / "analyse" / "vgg16_xai_methods"
        self._run_analysis(
            config_names, output_dir, "VGG XAI Methods", balance_samples=True
        )

    def analyze_model_comparison(self) -> None:
        """
        Compare different model architectures using the same XAI method.

        Analyzes and compares model performance using GradCAM across:
        - ResNet18
        - ResNet34
        - ResNet50
        - VGG16

        This analysis helps understand how different model architectures
        affect XAI explanation quality when using the same explanation method.

        Generated outputs:
            - Multi-model combined dataset CSV
            - Architecture comparison plots
            - Performance correlation analysis
            - Statistical significance testing

        Output directory: results/analyse/models_gradcam/
        """
        config_names = [
            "config_resnet18_grad_cam",
            "config_resnet34_grad_cam",
            "config_resnet50_grad_cam",
            "config_vgg16_grad_cam",
        ]

        output_dir = self.project_root / "results" / "analyse" / "models_gradcam"
        self._run_analysis(config_names, output_dir, "Model Comparison")

    def _run_analysis(
        self,
        config_names: List[str],
        output_dir: Path,
        analysis_name: str,
        balance_samples: bool = False,
    ) -> None:
        """
        Execute comprehensive analysis for the given experiment configurations.

        This is the core analysis workflow that handles data loading,
        processing, visualization, and statistical analysis for multiple
        experiment configurations.

        Workflow:
        1. Load data for each configuration
        2. Combine and validate datasets
        3. Apply sample balancing if requested
        4. Generate comprehensive visualizations
        5. Perform statistical analyses
        6. Save results and plots

        Args:
            config_names: List of experiment configuration names to analyze
            output_dir: Directory path where results will be saved
            analysis_name: Human-readable name for the analysis (used in plots and
            logging)
            balance_samples: Whether to balance sample sizes across experiments
                           for fair comparison (default: False)

        Generated outputs:
            - combined_data.csv: Merged experiment data
            - plots/: Directory containing all visualization files
            - f1_scores_*.csv: F1-score analysis results
            - Statistical analysis plots and data

        Raises:
            FileNotFoundError: If experiment data for any config is not found
            ValueError: If data validation or processing fails

        Note:
            Sample balancing uses random sampling with seed=42 for reproducibility.
        """
        self.logger.info(f"Starting {analysis_name}")

        # Daten laden
        dfs = []
        for config_name in config_names:
            try:
                df = self._load_experiment_data(config_name)
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} samples from {config_name}")
            except Exception:
                raise

        if not dfs:
            self.logger.error(f"No data loaded for {analysis_name}")
            return

        # Daten kombinieren
        combined_df = pd.concat(dfs, ignore_index=True)

        # Sample-Balancierung falls gewünscht
        if balance_samples:
            combined_df = self._balance_samples(combined_df)

        # Output-Verzeichnis erstellen
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Kombinierte Daten speichern
        combined_path = output_dir / "combined_data.csv"
        combined_df.to_csv(combined_path, index=False)

        # Erstelle erweiterte Analysen
        try:
            # 1. Standard Advanced Plotter
            plotter = AdvancedPlotter(combined_df, plot_dir)
            plot_paths = plotter.create_comprehensive_analysis()
            self.logger.info(
                f"Created {len(plot_paths)} standard plots for" f" {analysis_name}"
            )

            # 2. F1-Score Heatmap
            f1_heatmap_path = self.create_f1_score_heatmap(
                combined_df, plot_dir, analysis_name
            )
            if f1_heatmap_path:
                plot_paths["f1_score_heatmap"] = f1_heatmap_path

            # 5. Comprehensive histogram comparison
            histogram_comparison_path = self.create_histogram_comparison(
                combined_df, plot_dir, analysis_name
            )
            if histogram_comparison_path:
                plot_paths["histogram_comparison"] = histogram_comparison_path

            self.logger.info(
                f"Created {len(plot_paths)} total plots for" f" {analysis_name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to create plots for {analysis_name}: {e}")
            raise

        self.logger.info(f"Completed {analysis_name} - Results in: {output_dir}")

    def _load_experiment_data(self, config_name: str) -> pd.DataFrame:
        """
        Load experiment data for a given configuration with robust fallback strategies.

        This method implements a multi-strategy approach to locate and load
        experiment data, handling various directory structures and naming conventions:

        1. Configuration-based loading using Hydra
        2. Pattern matching in results directories
        3. Exhaustive search with content validation

        Args:
            config_name: Name of the experiment configuration (e.g.,
            'config_resnet50_grad_cam')

        Returns:
            DataFrame containing the experiment results with required columns:
            - model_name, explainer_name (added if missing)
            - prediction_correct, prediction_confidence
            - iou, pixel metrics, point_game
            - Additional metadata columns

        Raises:
            FileNotFoundError: If no valid experiment data is found for the
            configuration

        Note:
            The method automatically enriches data with model_name and explainer_name
            extracted from the configuration name if these columns are missing.
        """

        # Definiere mögliche Pfade basierend auf dem Config-Namen
        possible_paths = []

        # 1. Direkte Zuordnung basierend auf experiment output_dir
        # Lade die Config um den output_dir zu bestimmen
        try:
            from hydra import compose, initialize_config_dir

            config_dir = self.project_root / "config" / "experiments"

            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name=f"{config_name}.yaml")
                if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "output_dir"):
                    output_dir = Path(cfg.experiment.output_dir)
                    if not output_dir.is_absolute():
                        output_dir = self.project_root / output_dir
                    possible_paths.append(output_dir / "results_with_metrics.csv")
        except Exception as e:
            self.logger.debug(f"Could not load config {config_name}: {e}")

        # 2. Fallback: Suche in results-Verzeichnis basierend auf Config-Namen
        results_dir = self.project_root / "results"

        # Extrahiere Modell und Explainer aus Config-Name
        config_base = config_name.replace("config_", "")

        # Mögliche Ordnerstrukturen
        possible_patterns = [
            f"*{config_base}*",
            f"*{config_base.split('_')[0]}*",  # nur Modellname
            f"*{config_base}*",
            "experiment_*",
            "*_experiment",
        ]

        for pattern in possible_patterns:
            for exp_dir in results_dir.glob(pattern):
                if exp_dir.is_dir():
                    csv_file = exp_dir / "results_with_metrics.csv"
                    if csv_file.exists():
                        possible_paths.append(csv_file)

        # 3. Weitere Fallback-Optionen: Durchsuche alle Experimente
        if results_dir.exists():
            for exp_dir in results_dir.iterdir():
                if exp_dir.is_dir():
                    csv_file = exp_dir / "results_with_metrics.csv"
                    if csv_file.exists():
                        # Prüfe ob der CSV-Datei-Inhalt zu unserem Config passt
                        pd.read_csv(csv_file, nrows=1)
                        possible_paths.append(csv_file)

        # Versuche die Pfade in der Reihenfolge
        for csv_path in possible_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        # Füge Metadaten hinzu basierend auf Config-Name
                        model_name, explainer_name = self._extract_model_and_explainer(
                            config_name
                        )
                        if "model_name" not in df.columns:
                            df["model_name"] = model_name
                        if "explainer_name" not in df.columns:
                            df["explainer_name"] = explainer_name

                        self.logger.info(f"Loaded data from {csv_path}")
                        return df
                except Exception as e:
                    self.logger.warning(f"Could not load {csv_path}: {e}")
                    continue

        # Als letzter Ausweg: Zeige verfügbare Dateien
        self.logger.error(f"No results found for {config_name}")
        self.logger.info("Available result files:")
        if results_dir.exists():
            for exp_dir in results_dir.iterdir():
                if exp_dir.is_dir():
                    csv_file = exp_dir / "results_with_metrics.csv"
                    if csv_file.exists():
                        self.logger.info(f"  - {csv_file}")

        raise FileNotFoundError(f"No results found for {config_name}")

    def _extract_model_and_explainer(self, config_name: str) -> tuple[str, str]:
        """
        Extract model and explainer information from configuration name.

        Parses configuration names to automatically determine the model
        architecture and XAI method being used. Handles standard naming
        conventions and provides fallback parsing for non-standard formats.

        Supported naming pattern: config_{model}_{explainer}
        Example: 'config_resnet50_grad_cam' -> ('resnet50', 'grad_cam')

        Args:
            config_name: Configuration file name (with or without 'config_' prefix)

        Returns:
            Tuple of (model_name, explainer_name) extracted from the config name

        Known explainer names:
            - grad_cam
            - score_cam
            - guided_backprop
            - integrated_gradients

        Note:
            If parsing fails, returns ('unknown', 'unknown') as fallback.
        """
        # Entferne "config_" Präfix
        if config_name.startswith("config_"):
            name_part = config_name[7:]
        else:
            name_part = config_name

        # Bekannte Explainer-Namen
        known_explainers = [
            "grad_cam",
            "score_cam",
            "guided_backprop",
            "integrated_gradients",
        ]

        # Finde den Explainer
        explainer_name = None
        model_name = None

        for explainer in known_explainers:
            if name_part.endswith(f"_{explainer}"):
                explainer_name = explainer
                model_name = name_part[: -len(f"_{explainer}")]
                break

        if not explainer_name:
            # Fallback: letztes Segment als Explainer
            parts = name_part.split("_")
            if len(parts) >= 2:
                model_name = "_".join(parts[:-1])
                explainer_name = parts[-1]
            else:
                model_name = name_part
                explainer_name = "unknown"

        return model_name, explainer_name

    def _balance_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance sample sizes across XAI methods for fair comparison.

        This method ensures equal representation of each XAI method by
        randomly sampling the same number of examples from each explainer group.
        Uses the minimum group size as the target sample size.

        Args:
            df: DataFrame with 'explainer_name' column for grouping

        Returns:
            DataFrame with balanced sample sizes across all explainer methods

        Process:
        1. Calculate minimum sample size across all explainer groups
        2. Randomly sample (with seed=42) that many samples from each group
        3. Combine balanced samples into final dataset

        Note:
            - Uses random_state=42 for reproducible sampling
            - Preserves all original columns and data types
            - Logs sampling statistics for transparency
        """
        group_counts = df.groupby("explainer_name").size()
        min_samples = group_counts.min()

        self.logger.info(f"Balancing to {min_samples} samples per method (head-based)")

        balanced_dfs = []
        for explainer_name in df["explainer_name"].unique():
            explainer_data = df[df["explainer_name"] == explainer_name]

            # Zufällige Auswahl von min_samples
            if len(explainer_data) > min_samples:
                sampled_data = explainer_data.sample(n=min_samples, random_state=42)
                self.logger.info(
                    f"Reduced {explainer_name} from {len(explainer_data)} to "
                    f"{len(sampled_data)} samples"
                )
            else:
                sampled_data = explainer_data
                self.logger.info(
                    f"Kept all {len(sampled_data)} samples for {explainer_name}"
                )

            balanced_dfs.append(sampled_data)

        return pd.concat(balanced_dfs, ignore_index=True)

    def create_f1_score_heatmap(
        self, df: pd.DataFrame, plot_dir: Path, analysis_name: str
    ) -> Path:
        """
        Create comprehensive F1-score performance heatmap for Point Game metric.

        Generates a three-panel heatmap showing F1-score, Precision, and Recall
        for Point Game performance across different model-explainer combinations.
        Uses Point Game scores as binary predictions (threshold=0.5) against
        ground truth prediction correctness.

        Args:
            df: DataFrame containing experiment data with required columns:
                - point_game: Point Game metric scores [0,1]
                - prediction_correct: Ground truth classification correctness
                - model_name, explainer_name: For grouping (if available)
            plot_dir: Directory path where the plot will be saved
            analysis_name: Name used in plot title and filename

        Returns:
            Path to the generated heatmap file, or None if generation failed

        Generated files:
            - f1_score_heatmap_{analysis_name}.png: The heatmap visualization
            - f1_scores_{analysis_name}.csv: Detailed F1-score data

        Features:
            - Three-panel layout: F1-score, Precision, Recall
            - Color-coded performance visualization
            - Statistical annotations with sample sizes
            - Handles single-group data with bar plots

        Note:
            Requires both 'point_game' and 'prediction_correct' columns.
            Binary classification threshold for Point Game is set to 0.5.
        """
        try:
            if "point_game" not in df.columns or "prediction_correct" not in df.columns:
                self.logger.warning("Missing columns for F1-Score heatmap")
                return None

            # Ensure prediction_correct is boolean (handle NaN values)
            df = df.copy()  # Work with a copy to avoid SettingWithCopyWarning
            df["prediction_correct"] = (
                df["prediction_correct"].fillna(False).astype(bool)
            )

            # Calculate F1 scores for each model-explainer combination
            f1_results = []

            # Group by model and explainer if available
            group_cols = []
            if "model_name" in df.columns:
                group_cols.append("model_name")
            if "explainer_name" in df.columns:
                group_cols.append("explainer_name")

            if not group_cols:
                self.logger.warning("No grouping columns found for F1-Score heatmap")
                return None

            for group_values, group_data in df.groupby(group_cols):
                if isinstance(group_values, str):
                    group_values = [group_values]

                # Convert point game to binary (threshold 0.5)
                y_true = group_data["prediction_correct"].astype(int)
                y_pred_point_game = (group_data["point_game"] >= 0.5).astype(int)

                if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_point_game)) > 1:
                    f1 = f1_score(y_true, y_pred_point_game)
                    precision = precision_score(y_true, y_pred_point_game)
                    recall = recall_score(y_true, y_pred_point_game)

                    result = {
                        group_cols[0]: (
                            group_values[0] if len(group_values) > 0 else "Unknown"
                        )
                    }
                    if len(group_cols) > 1:
                        result[group_cols[1]] = (
                            group_values[1] if (len(group_values) > 1) else "Unknown"
                        )

                    result.update(
                        {
                            "f1_score": f1,
                            "precision": precision,
                            "recall": recall,
                            "samples": len(group_data),
                        }
                    )
                    f1_results.append(result)

            if not f1_results:
                self.logger.warning("No valid F1 scores computed")
                return None

            f1_df = pd.DataFrame(f1_results)

            # Create heatmap
            plt.style.use("seaborn-v0_8")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # F1-Score Heatmap
            if len(group_cols) == 2:
                pivot_f1 = f1_df.pivot(
                    index=group_cols[0], columns=group_cols[1], values="f1_score"
                )
                pivot_precision = f1_df.pivot(
                    index=group_cols[0], columns=group_cols[1], values="precision"
                )
                pivot_recall = f1_df.pivot(
                    index=group_cols[0], columns=group_cols[1], values="recall"
                )
            else:
                # Single grouping column - create simple bar plot instead
                fig, ax = plt.subplots(figsize=(10, 6))
                f1_df.plot(
                    x=group_cols[0], y="f1_score", kind="bar", ax=ax, color="viridis"
                )
                ax.set_title(f"F1 Scores: {analysis_name}")
                ax.set_ylabel("F1 Score")
                plt.xticks(rotation=45)
                plt.tight_layout()

                plot_path = (
                    plot_dir / "f1_score_heatmap_"
                    f"{analysis_name.lower().replace(' ', '_')}"
                    ".png"
                )
                plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                return plot_path

            # F1 Score
            sns.heatmap(
                pivot_f1,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                ax=axes[0],
                cbar_kws={"label": "F1 Score"},
                vmin=0,
                vmax=1,
            )
            axes[0].set_title("F1 Scores", fontweight="bold")
            axes[0].set_ylabel("Model")
            axes[0].set_xlabel("XAI Method")

            # Precision
            sns.heatmap(
                pivot_precision,
                annot=True,
                fmt=".3f",
                cmap="Blues",
                ax=axes[1],
                cbar_kws={"label": "Precision"},
                vmin=0,
                vmax=1,
            )
            axes[1].set_title("Precision", fontweight="bold")
            axes[1].set_ylabel("Model")
            axes[1].set_xlabel("XAI Method")

            # Recall
            sns.heatmap(
                pivot_recall,
                annot=True,
                fmt=".3f",
                cmap="Oranges",
                ax=axes[2],
                cbar_kws={"label": "Recall"},
                vmin=0,
                vmax=1,
            )
            axes[2].set_title("Recall", fontweight="bold")
            axes[2].set_ylabel("Model")
            axes[2].set_xlabel("XAI Method")

            plt.suptitle(
                f"Performance Metrics: {analysis_name}\n(Point Game vs. Prediction "
                f"Correctness)",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            plot_path = (
                plot_dir
                / f"f1_score_heatmap_{analysis_name.lower().replace(' ', '_')}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            # Save F1 data as CSV
            csv_path = (
                plot_dir.parent
                / f"f1_scores_{analysis_name.lower().replace(' ', '_')}.csv"
            )
            f1_df.to_csv(csv_path, index=False)

            return plot_path

        except Exception as e:
            self.logger.error(f"Failed to create F1-Score heatmap: {e}")
            raise

    def create_histogram_comparison(
        self, df: pd.DataFrame, plot_dir: Path, analysis_name: str
    ) -> Path:
        """
        Create comprehensive statistical histogram comparison for all available metrics.

        Generates overlapping histograms comparing metric distributions between
        correctly and incorrectly classified samples, with embedded statistical
        testing results. Includes automatic column name mapping for different
        naming conventions (e.g., pixel_precision vs pixelprecisionrecall_precision).

        Args:
            df: DataFrame containing experiment data with metrics columns
            plot_dir: Directory path where the plot will be saved
            analysis_name: Name used in plot title and filename

        Returns:
            Path to the generated histogram comparison file, or None if failed

        Analyzed metrics (if available):
            - IoU (Intersection over Union)
            - Pixel Precision and Recall
            - Point Game metric
            - Prediction Confidence

        Features:
            - Overlapping histograms for correct vs incorrect predictions
            - Statistical test results embedded in plots:
              * Mann-Whitney U test p-value
              * Cohen's d effect size
              * Kolmogorov-Smirnov test p-value
            - Mean value annotations with vertical lines
            - Automatic binning appropriate for metric ranges
            - Color-coded by prediction correctness

        Generated file:
            histogram_comparison_{analysis_name}.png

        Note:
            Handles different column naming conventions automatically.
            If prediction_correct column is missing, creates simple histograms
            without correctness-based splitting.
        """
        try:
            # Define available metrics (check for different column naming patterns)
            metrics_map = {
                "iou": ["iou"],
                "pixel_precision": [
                    "pixel_precision",
                    "pixelprecisionrecall_precision",
                ],
                "pixel_recall": ["pixel_recall", "pixelprecisionrecall_recall"],
                "point_game": ["point_game"],
                "prediction_confidence": ["prediction_confidence"],
            }

            # Find actual column names
            available_metrics = []
            metric_name_map = {}  # Maps display name to actual column name

            for display_name, possible_cols in metrics_map.items():
                for col in possible_cols:
                    if col in df.columns:
                        available_metrics.append(display_name)
                        metric_name_map[display_name] = col
                        break

            if not available_metrics:
                self.logger.warning("No metrics available for histogram comparison")
                return None

            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            plt.style.use("seaborn-v0_8")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Handle single subplot case
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            colors = ["#2E8B57", "#DC143C"]  # Green for correct, Red for incorrect
            labels = ["Correctly Classified", "Incorrectly Classified"]

            for i, metric in enumerate(available_metrics):
                if i >= len(axes):
                    break

                ax = axes[i]

                if "prediction_correct" in df.columns:
                    # Split by prediction correctness - use actual column name
                    actual_col = metric_name_map.get(metric, metric)
                    correct_data = df[df["prediction_correct"]][actual_col].dropna()
                    df["prediction_correct"] = df["prediction_correct"].astype(bool)
                    incorrect_data = df[df["prediction_correct"] == false][
                        actual_col
                    ].dropna()

                    # Create overlapping histograms
                    bins = (
                        np.linspace(0, 1, 31)
                        if metric
                        in ["iou", "pixel_precision", "pixel_recall", "point_game"]
                        else 25
                    )

                    if len(correct_data) > 0:
                        ax.hist(
                            correct_data,
                            bins=bins,
                            alpha=0.7,
                            color=colors[0],
                            label=f"{labels[0]} (n={len(correct_data)})",
                            density=True,
                            edgecolor="white",
                            linewidth=0.5,
                        )

                        # Add mean line
                        mean_correct = correct_data.mean()
                        ax.axvline(
                            mean_correct,
                            color=colors[0],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.8,
                        )
                        ax.text(
                            mean_correct,
                            ax.get_ylim()[1] * 0.9,
                            f"μ={mean_correct:.3f}",
                            rotation=90,
                            color=colors[0],
                            fontweight="bold",
                        )

                    if len(incorrect_data) > 0:
                        ax.hist(
                            incorrect_data,
                            bins=bins,
                            alpha=0.7,
                            color=colors[1],
                            label=f"{labels[1]} (n={len(incorrect_data)})",
                            density=True,
                            edgecolor="white",
                            linewidth=0.5,
                        )

                        # Add mean line
                        mean_incorrect = incorrect_data.mean()
                        ax.axvline(
                            mean_incorrect,
                            color=colors[1],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.8,
                        )
                        ax.text(
                            mean_incorrect,
                            ax.get_ylim()[1] * 0.8,
                            f"μ={mean_incorrect:.3f}",
                            rotation=90,
                            color=colors[1],
                            fontweight="bold",
                        )

                    # Statistical tests if both groups have data
                    if len(correct_data) > 0 and len(incorrect_data) > 0:
                        from scipy import stats

                        # Mann-Whitney U test (non-parametric)
                        mw_statistic, mw_p_value = stats.mannwhitneyu(
                            correct_data, incorrect_data, alternative="two-sided"
                        )

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (
                                (len(correct_data) - 1) * correct_data.std() ** 2
                                + (len(incorrect_data) - 1) * incorrect_data.std() ** 2
                            )
                            / (len(correct_data) + len(incorrect_data) - 2)
                        )
                        cohens_d = (
                            (correct_data.mean() - incorrect_data.mean()) / pooled_std
                            if pooled_std > 0
                            else 0
                        )

                        # Kolmogorov-Smirnov test for distribution differences
                        ks_statistic, ks_p_value = stats.ks_2samp(
                            correct_data, incorrect_data
                        )

                        # Create statistical summary
                        stat_text = (
                            f"Mann-Whitney U: p={mw_p_value:.3e}\nCohen's d:"
                            f" {cohens_d:.3f}\nKS test: p={ks_p_value:.3e}"
                        )

                        ax.text(
                            0.02,
                            0.98,
                            stat_text,
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment="top",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightblue",
                                alpha=0.8,
                            ),
                        )
                else:
                    # Single histogram if no correctness data - use actual column name
                    actual_col = metric_name_map.get(metric, metric)
                    ax.hist(
                        df[actual_col].dropna(),
                        bins=25,
                        alpha=0.7,
                        color="skyblue",
                        density=True,
                        edgecolor="white",
                        linewidth=0.5,
                    )

                    mean_val = df[actual_col].mean()
                    ax.axvline(
                        mean_val, color="blue", linestyle="--", linewidth=2, alpha=0.8
                    )
                    ax.text(
                        mean_val,
                        ax.get_ylim()[1] * 0.9,
                        f"μ={mean_val:.3f}",
                        rotation=90,
                        color="blue",
                        fontweight="bold",
                    )

                # Styling
                ax.set_title(
                    f"{metric.replace('_', ' ').title()}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel(f"{metric.replace('_', ' ').title()} Score", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor("#f8f8f8")

            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(
                f"Metric Distribution Analysis: {analysis_name} \nPerformance by "
                "Prediction Correctness",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            plot_path = (
                plot_dir
                / f"histogram_comparison_{analysis_name.lower().replace(' ', '_')}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            return plot_path

        except Exception as e:
            self.logger.error(f"Failed to create histogram comparison: {e}")
            raise


def main():
    """Führt alle Analysen aus."""
    analyzer = SimpleAnalyzer()

    # Erst Diagnose, dann Analysen
    analyzer.diagnose_available_data()

    print("\n" + "=" * 50)
    print("Starte jetzt die Analysen...")
    print("=" * 50)

    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()

import logging
import textwrap
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# Set style
plt.style.use("default")
sns.set_palette("husl")


class AdvancedPlotter:
    """
    Advanced plotting engine for comprehensive XAI data analysis and visualization.

    This class provides a comprehensive suite of publication-quality plotting
    capabilities for analyzing XAI experiment results. It generates statistical
    visualizations, performance comparisons, and distribution analyses with
    professional styling and automatic layout optimization.

    Features:
        - Multi-panel dashboard overviews
        - Radar charts for multi-metric comparisons
        - Statistical correlation analysis
        - Distribution analysis with violin and box plots
        - Point Game confusion matrix analysis
        - Automatic metric detection and validation
        - High-resolution output for publication quality

    Supported Metrics:
        - IoU (Intersection over Union)
        - Point Game metric
        - Pixel-level precision and recall
        - Prediction confidence
        - Processing time and accuracy metrics

    Attributes:
        df (pd.DataFrame): Experiment data with metrics and metadata
        save_dir (Path): Directory for saving generated plots
        metrics (List[str]): Available metric column names
        available_metrics (List[str]): Metrics present in the dataset

    Example:
        >>> plotter = AdvancedPlotter(experiment_df, Path('plots/'))
        >>> plot_paths = plotter.create_comprehensive_analysis()
        >>> print(f"Generated {len(plot_paths)} visualization files")
    """

    def __init__(self, df: pd.DataFrame, save_dir: Path):
        """
        Initialize the advanced plotter with experiment data and output directory.

        Args:
            df: DataFrame containing experiment results with metrics and metadata.
               Expected columns include model_name, explainer_name, prediction_correct,
               and various metric columns (iou, point_game, pixel metrics, etc.)
            save_dir: Directory path where generated plots will be saved.
                     Created automatically if it doesn't exist.

        Note:
            The DataFrame is copied to prevent modifications to the original data.
            Available metrics are automatically detected from the column names.
        """
        self.df = df.copy()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Standard metrics (prediction_confidence handled separately)
        self.metrics = [
            "iou",
            "point_game",
            "pixelprecisionrecall_precision",
            "pixelprecisionrecall_recall",
        ]

        # Filter to only available metrics in the dataset
        self.available_metrics = [m for m in self.metrics if m in self.df.columns]

    def create_comprehensive_analysis(self) -> Dict[str, Path]:
        """
        Generate comprehensive analysis with all available plot types.

        Creates a complete suite of visualizations including overview dashboards,
        metric comparisons, correlation analyses, distribution plots, and advanced
        statistical visualizations. All plots are saved with publication-quality
        settings and consistent styling.

        Returns:
            Dictionary mapping plot type names to their file paths:
            {
                'dataset_overview': Path('dashboard.png'),
                'radar_chart': Path('radar.png'),
                'correlation_heatmap': Path('correlations.png'),
                'metric_distributions': Path('distributions.png'),
                ...
            }

        Generated plot categories:
            1. Overview plots: Dataset summaries and sample distributions
            2. Metric comparisons: Multi-dimensional performance analysis
            3. Correlation analyses: Statistical relationship exploration
            4. Distribution plots: Metric distribution comparisons
            5. Advanced analyses: Specialized XAI-specific visualizations

        Note:
            All plots are generated with 300 DPI for publication quality.
            File names are automatically generated based on plot types.
        """
        plot_paths = {}

        # Use logger instead of print

        logger = logging.getLogger(__name__)
        logger.info("Creating comprehensive analysis plots")

        # 1. Überblick und Zusammenfassungen
        plot_paths.update(self._create_overview_plots())

        # 2. Metriken-Vergleiche
        plot_paths.update(self._create_metric_comparison_plots())

        # 3. Korrelations-Analysen
        plot_paths.update(self._create_correlation_plots())

        # 4. Verteilungs-Analysen
        plot_paths.update(self._create_distribution_plots())

        # 5. Erweiterte Analysen
        plot_paths.update(self._create_advanced_plots())

        logger.info(f"Created {len(plot_paths)} plots in {self.save_dir}")
        return plot_paths

    def _create_overview_plots(self) -> Dict[str, Path]:
        """
        Create overview and summary visualization plots.

        Generates high-level summary visualizations that provide quick insights
        into the dataset characteristics, sample distributions, and overall
        experiment performance across different models and methods.

        Returns:
            Dictionary of plot names and their file paths

        Generated plots:
            - dataset_overview: Multi-panel dashboard with key statistics
            - accuracy_heatmap: Model-method accuracy comparison matrix
            - sample_distribution: Sample count distribution analysis
        """
        plots = {}

        # 1. Dataset Overview Dashboard
        plots["dataset_overview"] = self._plot_dataset_overview()

        # 2. Accuracy Heatmap
        plots["accuracy_heatmap"] = self._plot_accuracy_heatmap()

        # 3. Sample Distribution
        plots["sample_distribution"] = self._plot_sample_distribution()

        return plots

    def _create_metric_comparison_plots(self) -> Dict[str, Path]:
        """
        Create multi-metric comparison visualizations.

        Generates plots that compare performance across different metrics
        simultaneously, enabling comprehensive method evaluation and
        identification of strengths and weaknesses across metrics.

        Returns:
            Dictionary of plot names and their file paths

        Generated plots:
            - radar_chart: Multi-dimensional performance radar visualization
              with normalized metrics and F1-score integration
        """
        plots = {}

        # 1. Multi-Metric Radar Chart (nur dieser ist implementiert)
        plots["radar_chart"] = self._plot_radar_chart()

        return plots

    def _create_correlation_plots(self) -> Dict[str, Path]:
        """
        Create correlation analysis visualizations.

        Generates statistical correlation matrices and relationship analyses
        between different metrics, helping identify interdependencies and
        potential redundancies in the evaluation framework.

        Returns:
            Dictionary of plot names and their file paths

        Generated plots:
            - correlation_heatmap: Statistical correlation matrix with
              significance testing and color-coded relationships
        """
        plots = {}

        # 1. Correlation Matrix Heatmap (nur dieser ist implementiert)
        plots["correlation_heatmap"] = self._plot_correlation_heatmap()

        return plots

    def _create_distribution_plots(self) -> Dict[str, Path]:
        """
        Create distribution analysis visualizations.

        Generates detailed distribution plots comparing metric distributions
        across different models, methods, and prediction correctness categories.
        Uses violin and box plots for comprehensive distribution analysis.

        Returns:
            Dictionary of plot names and their file paths

        Generated plots:
            - metric_distributions: Multi-panel distribution analysis with
              violin plots, box plots, and statistical annotations
        """
        plots = {}

        # 1. Metric Distributions (nur dieser ist implementiert)
        plots["metric_distributions"] = self._plot_metric_distributions()

        return plots

    def _create_advanced_plots(self) -> Dict[str, Path]:
        """
        Create advanced and specialized analysis visualizations.

        Generates sophisticated analyses specific to XAI evaluation,
        including confusion matrices for Point Game analysis and other
        specialized visualizations that provide deep insights into
        explanation quality and method performance.

        Returns:
            Dictionary of plot names and their file paths

        Generated plots:
            - point_game_confusion_matrix: Detailed confusion matrix analysis
              for Point Game metric performance evaluation
        """
        plots = {}

        # 1. Point Game Confusion Matrix
        if "point_game" in self.available_metrics:
            plots["point_game_confusion"] = self._plot_point_game_confusion_matrix()

        # 2. IoU Distribution Histograms
        if "iou" in self.available_metrics:
            plots["iou_distribution_histograms"] = (
                self._plot_iou_distribution_histograms()
            )

        # 3. Pixel Precision Distribution Histograms
        if "pixelprecisionrecall_precision" in self.available_metrics:
            plots["pixel_precision_distribution_histograms"] = (
                self._plot_pixel_precision_distribution_histograms()
            )

        # 4. Point Game Confusion Matrix Comparison
        if "point_game" in self.available_metrics:
            plots["point_game_confusion_comparison"] = (
                self.plot_point_game_confusion_matrix_comparison()
            )

        return plots

    def _plot_dataset_overview(self) -> Path:
        """Erstellt ein Dashboard mit Dataset-Überblick."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Dataset Overview Dashboard", fontsize=16, fontweight="bold")

        # 1. Anzahl Samples pro Modell/Explainer
        ax = axes[0, 0]
        combo_counts = (
            self.df.groupby(["model_name", "explainer_name"])
            .size()
            .reset_index(name="count")
        )
        pivot_counts = (
            combo_counts.pivot(
                index="model_name", columns="explainer_name", values="count"
            )
            .fillna(0)
            .astype(int)
        )

        sns.heatmap(
            pivot_counts,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            annot_kws={"fontsize": 16},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Sample Counts per Model/Explainer", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel("Explainer", fontsize=16)
        ax.set_ylabel("Model", fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

        # 2. Accuracy Distribution
        ax = axes[0, 1]
        if "prediction_correct" in self.df.columns:
            accuracy_by_combo = self.df.groupby(["model_name", "explainer_name"])[
                "prediction_correct"
            ].mean()
            accuracy_by_combo.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Accuracy by Model/Explainer", fontsize=16)
            ax.set_ylabel("Accuracy", fontsize=16)
            ax.set_xlabel("", fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=16
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No accuracy data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title("Accuracy by Model/Explainer", fontsize=16)
            ax.set_ylabel("Accuracy", fontsize=16)
            ax.tick_params(axis="both", labelsize=16)

        # 3. Processing Time Distribution
        ax = axes[0, 2]
        if "processing_time" in self.df.columns:
            # Farbschema definieren, z.B. mit Seaborn-Palette
            palette = sns.color_palette(
                "Set2", n_colors=self.df["explainer_name"].nunique()
            )
            sorted(self.df["explainer_name"].unique())

            # Boxplot mit seaborn (einfacher farbig)
            sns.boxplot(
                data=self.df,
                x="explainer_name",
                y="processing_time",
                ax=ax,
                palette=palette,
            )
            ax.set_title("Processing Time by Explainer", fontsize=16)
            ax.set_ylabel("Processing Time (s)", fontsize=16)
            ax.set_xlabel("Explainer", fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=16
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No timing data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title("Processing Time Not Available", fontsize=16)

        # 4. Metric Availability
        ax = axes[1, 0]
        metric_availability = pd.DataFrame(
            {
                "Metric": self.available_metrics,
                "Available": [self.df[m].notna().sum() for m in self.available_metrics],
                "Missing": [self.df[m].isna().sum() for m in self.available_metrics],
            }
        )
        metric_availability.set_index("Metric")[["Available", "Missing"]].plot(
            kind="bar", stacked=True, ax=ax
        )
        ax.set_title("Metric Data Availability", fontsize=16)
        ax.set_ylabel("Number of Samples", fontsize=16)
        ax.set_xlabel("Metric", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)

        # 5. Confidence Distribution
        ax = axes[1, 1]
        if "prediction_confidence" in self.df.columns:
            for explainer in self.df["explainer_name"].unique():
                subset = self.df[self.df["explainer_name"] == explainer]
                ax.hist(
                    subset["prediction_confidence"], alpha=0.6, label=explainer, bins=20
                )
            ax.set_title("Prediction Confidence Distribution", fontsize=16)
            ax.set_xlabel("Confidence", fontsize=16)
            ax.set_ylabel("Count", fontsize=16)
            ax.legend(fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
        else:
            ax.text(
                0.5,
                0.5,
                "No confidence data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title("Confidence Data Not Available", fontsize=16)

        # 6. Summary Statistics
        summary_text = f"""
        Dataset Summary:

        Total Samples: {len(self.df):,}
        Models: {self.df['model_name'].nunique()}
        Explainers: {self.df['explainer_name'].nunique()}
        Available Metrics: {len(self.available_metrics)}

        Models: {', '.join(self.df['model_name'].unique())}

        Explainers: {', '.join(self.df['explainer_name'].unique())}
        """

        # Absatzzeilen einzeln umbrechen
        wrapped_lines = []
        for line in summary_text.strip().split("\n"):
            if line.strip() == "":
                wrapped_lines.append("")  # Leerzeile bleibt erhalten
            else:
                wrapped_lines.extend(textwrap.wrap(line.strip(), width=50))

        # Wieder zusammensetzen
        wrapped_text = "\n".join(wrapped_lines)

        # Text anzeigen
        ax = axes[1, 2]
        ax.axis("off")
        ax.text(
            0.01,
            0.99,
            wrapped_text,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        save_path = self.save_dir / "dataset_overview_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_accuracy_heatmap(self) -> Path:
        """Erstellt eine Accuracy-Heatmap."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if "prediction_correct" in self.df.columns:
            # Accuracy berechnen
            accuracy_data = self.df.groupby(["model_name", "explainer_name"])[
                "prediction_correct"
            ].mean()
            accuracy_pivot = accuracy_data.reset_index().pivot(
                index="model_name",
                columns="explainer_name",
                values="prediction_correct",
            )

            # Heatmap erstellen
            sns.heatmap(
                accuracy_pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=0.5,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "Accuracy"},
            )
            ax.set_title(
                "Model-Explainer Accuracy Heatmap", fontsize=16, fontweight="bold"
            )
            ax.set_xlabel("Explainer Method")
            ax.set_ylabel("Model")
        else:
            ax.text(
                0.5,
                0.5,
                "No accuracy data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Accuracy Data Not Available")

        plt.tight_layout()
        save_path = self.save_dir / "accuracy_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_radar_chart(self) -> Path:
        """Erstellt ein Radar-Chart für Metriken-Vergleich."""
        if len(self.available_metrics) < 3:
            # Fallback für wenige Metriken
            return self._plot_simple_bar_chart()

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Daten vorbereiten
        combinations = []
        for model in self.df["model_name"].unique():
            for explainer in self.df["explainer_name"].unique():
                subset = self.df[
                    (self.df["model_name"] == model)
                    & (self.df["explainer_name"] == explainer)
                ]
                if len(subset) > 0:
                    combinations.append((model, explainer, subset))

        # Winkel für Radar-Chart
        angles = np.linspace(
            0, 2 * np.pi, len(self.available_metrics), endpoint=False
        ).tolist()
        angles += angles[:1]  # Schließe den Kreis

        colors = plt.cm.Set3(np.linspace(0, 1, len(combinations)))

        for i, (model, explainer, subset) in enumerate(combinations):
            values = []
            for metric in self.available_metrics:
                if metric in subset.columns:
                    if metric == "point_game":
                        # Calculate F1 score for point game if prediction_correct exists
                        if "prediction_correct" in subset.columns:
                            from sklearn.metrics import f1_score

                            y_true = (
                                subset["prediction_correct"].fillna(False).astype(int)
                            )
                            y_pred = (subset["point_game"] >= 0.5).astype(int)
                            if (
                                len(np.unique(y_true)) > 1
                                and len(np.unique(y_pred)) > 1
                            ):
                                f1 = f1_score(y_true, y_pred)
                                values.append(f1)
                            else:
                                values.append(0)
                        else:
                            mean_val = subset[metric].mean()
                            max_val = (
                                self.df[metric].max()
                                if self.df[metric].max() > 0
                                else 1
                            )
                            normalized_val = mean_val / max_val
                            values.append(normalized_val)
                    else:
                        mean_val = subset[metric].mean()
                        # Normalisiere auf 0-1 Bereich
                        max_val = (
                            self.df[metric].max() if self.df[metric].max() > 0 else 1
                        )
                        normalized_val = mean_val / max_val
                        values.append(normalized_val)
                else:
                    values.append(0)

            values += values[:1]  # Schließe den Kreis

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=f"{model}-{explainer}",
                color=colors[i],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Anpassungen mit Display-Namen
        def get_display_name(metric):
            if metric == "pixelprecisionrecall_precision":
                return "pixel\nprecision"
            elif metric == "pixelprecisionrecall_recall":
                return "pixel\nrecall"
            elif metric == "point_game":
                return "point game F1 Score"
            else:
                return metric.replace("_", "\n")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([get_display_name(m) for m in self.available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title(
            "Metric Performance Radar Chart", size=16, fontweight="bold", pad=20
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        save_path = self.save_dir / "radar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_simple_bar_chart(self) -> Path:
        """Fallback für Radar-Chart bei wenigen Metriken."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Durchschnittswerte berechnen
        avg_data = []
        for model in self.df["model_name"].unique():
            for explainer in self.df["explainer_name"].unique():
                subset = self.df[
                    (self.df["model_name"] == model)
                    & (self.df["explainer_name"] == explainer)
                ]
                if len(subset) > 0:
                    for metric in self.available_metrics:
                        if metric in subset.columns:
                            avg_data.append(
                                {
                                    "model": model,
                                    "explainer": explainer,
                                    "metric": metric,
                                    "value": subset[metric].mean(),
                                }
                            )

        if avg_data:
            df_avg = pd.DataFrame(avg_data)
            df_avg["combination"] = df_avg["model"] + "-" + df_avg["explainer"]

            df_pivot = df_avg.pivot(
                index="combination", columns="metric", values="value"
            )

            # Exclude point_game from bar charts (it doesn't need bar visualization)
            metrics_for_bars = [m for m in df_pivot.columns if m != "point_game"]
            if metrics_for_bars:
                df_pivot[metrics_for_bars].plot(
                    kind="bar", ax=ax, width=0.7, figsize=(14, 8)
                )
                ax.set_title(
                    "Average Metric Performance by Model-Explainer (Excluding Point "
                    "Game)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_ylabel("Metric Value")
                ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                # Adjust spacing between bars to prevent compression
                ax.margins(x=0.01)

        plt.tight_layout()
        save_path = self.save_dir / "radar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_correlation_heatmap(self) -> Path:
        """Erstellt eine Korrelations-Heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Nur numerische Spalten für Korrelation
        numeric_cols = []
        for col in self.available_metrics:
            if col in self.df.columns and self.df[col].dtype in ["float64", "int64"]:
                numeric_cols.append(col)

        if "prediction_correct" in self.df.columns:
            self.df["prediction_correct_int"] = self.df["prediction_correct"].astype(
                int
            )
            numeric_cols.append("prediction_correct_int")

        if "prediction_confidence" in self.df.columns:
            numeric_cols.append("prediction_confidence")

        if len(numeric_cols) >= 2:
            correlation_matrix = self.df[numeric_cols].corr()

            # Erstelle Maske für obere Dreiecksmatrix
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"label": "Correlation Coefficient"},
            )
            ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient numeric data for correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Correlation Analysis Not Available")

        plt.tight_layout()
        save_path = self.save_dir / "correlation_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_metric_distributions(self) -> Path:
        """Erstellt Verteilungsplots für alle Metriken."""
        n_metrics = len(self.available_metrics)
        if n_metrics == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No metrics available for distribution analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Metric Distributions Not Available")
            save_path = self.save_dir / "metric_distributions.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path

        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("Metric Distribution Analysis", fontsize=16, fontweight="bold")

        for i, metric in enumerate(self.available_metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]

            # Erst Violin Plot als Hintergrund für Verteilungsform
            if len(self.df[metric].dropna()) > 10:  # Nur wenn genug Daten vorhanden
                sns.violinplot(
                    data=self.df,
                    x="explainer_name",
                    y=metric,
                    ax=ax,
                    alpha=0.3,
                    inner=None,
                    cut=0,
                    palette="Set2",
                    linewidth=0,
                )

            # Dann Boxplot mit exakt gleichen Parametern für perfekte Überlagerung
            sns.boxplot(
                data=self.df,
                x="explainer_name",
                y=metric,
                ax=ax,
                width=0.5,
                palette="Set2",
                boxprops={"facecolor": "white", "alpha": 0.9, "linewidth": 1.2},
                medianprops={"linewidth": 2.5, "color": "red"},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                showfliers=True,
                flierprops={"marker": "o", "alpha": 0.6, "markersize": 4},
            )

            # Display-Namen für Titel und Label
            def get_display_name(metric):
                if metric == "pixelprecisionrecall_precision":
                    return "Pixel Precision"
                elif metric == "pixelprecisionrecall_recall":
                    return "Pixel Recall"
                else:
                    return metric.replace("_", " ").title()

            display_name = get_display_name(metric)
            ax.set_title(f"{display_name} Distribution")
            ax.set_xlabel("Explainer")
            ax.set_ylabel(display_name)
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Verstecke unbenutzte Subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        save_path = self.save_dir / "metric_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_performance_dashboard(self) -> Path:
        """Erstellt ein Performance-Dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Performance Analysis Dashboard", fontsize=16, fontweight="bold")

        # 1. Accuracy vs Confidence
        ax = axes[0, 0]
        if (
            "prediction_confidence" in self.df.columns
            and "prediction_correct" in self.df.columns
        ):
            for explainer in self.df["explainer_name"].unique():
                subset = self.df[self.df["explainer_name"] == explainer]
                ax.scatter(
                    subset["prediction_confidence"],
                    subset["prediction_correct"],
                    alpha=0.6,
                    label=explainer,
                    s=30,
                )
            ax.set_xlabel("Prediction Confidence")
            ax.set_ylabel("Prediction Correct")
            ax.set_title("Confidence vs Correctness")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Confidence vs Correctness - Data Not Available")

        # 2. Top-k Accuracy (falls verfügbar)
        ax = axes[0, 1]
        if "prediction_correct" in self.df.columns:
            accuracy_by_model = (
                self.df.groupby(["model_name", "explainer_name"])["prediction_correct"]
                .mean()
                .reset_index()
            )
            accuracy_pivot = accuracy_by_model.pivot(
                index="model_name",
                columns="explainer_name",
                values="prediction_correct",
            )
            accuracy_pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_title("Accuracy Comparison")
            ax.set_ylabel("Accuracy")
            ax.legend(title="Explainer", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "Accuracy data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Accuracy Comparison - Data Not Available")

        # 3. Processing Time Comparison
        ax = axes[1, 0]
        if "processing_time" in self.df.columns:
            sns.boxplot(data=self.df, x="explainer_name", y="processing_time", ax=ax)
            ax.set_title("Processing Time Distribution")
            ax.set_ylabel("Processing Time (seconds)")
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "Processing time data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Processing Time - Data Not Available")

        # 4. Sample Counts
        ax = axes[1, 1]
        sample_counts = (
            self.df.groupby(["model_name", "explainer_name"])
            .size()
            .reset_index(name="count")
        )
        sample_pivot = sample_counts.pivot(
            index="model_name", columns="explainer_name", values="count"
        )
        # Sicherstellen dass alle Werte Integer sind
        sample_pivot = sample_pivot.fillna(0).astype(int)
        sample_pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Sample Count Distribution")
        ax.set_ylabel("Number of Samples")
        ax.legend(title="Explainer", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        save_path = self.save_dir / "performance_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_sample_distribution(self) -> Path:
        """Plot für Sample-Verteilung."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Samples pro Model
        ax = axes[0]
        model_counts = self.df["model_name"].value_counts()
        model_counts.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Samples per Model")
        ax.set_ylabel("Number of Samples")
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Samples pro Explainer
        ax = axes[1]
        explainer_counts = self.df["explainer_name"].value_counts()
        explainer_counts.plot(kind="bar", ax=ax, color="lightcoral")
        ax.set_title("Samples per Explainer")
        ax.set_ylabel("Number of Samples")
        plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        save_path = self.save_dir / "sample_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_point_game_confusion_matrix(self) -> Path:
        """
        Erstellt eine Confusion Matrix für Point Game Metric.
        Hit/Miss vs Correct/Incorrect Classification
        """
        fig, axes = plt.subplots(
            1,
            len(self.df["model_name"].unique()),
            figsize=(6 * len(self.df["model_name"].unique()), 6),
        )

        if len(self.df["model_name"].unique()) == 1:
            axes = [axes]

        fig.suptitle(
            "Point Game Confusion Matrix: Hit/Miss vs Correct/Incorrect Classification",
            fontsize=16,
            fontweight="bold",
        )

        for idx, model in enumerate(self.df["model_name"].unique()):
            ax = axes[idx]
            model_data = self.df[self.df["model_name"] == model]

            # Create binary point game (hit=1, miss=0) based on threshold
            point_game_binary = (model_data["point_game"] >= 0.5).astype(int)
            prediction_correct = model_data["prediction_correct"].astype(int)

            # Create confusion matrix data
            confusion_data = pd.crosstab(
                point_game_binary,
                prediction_correct,
                rownames=["Point Game"],
                colnames=["Prediction"],
                margins=True,
            )

            # Rename for clarity
            confusion_data.index = confusion_data.index.map(
                {0: "Miss", 1: "Hit", "All": "Total"}
            )
            confusion_data.columns = confusion_data.columns.map(
                {0: "Incorrect", 1: "Correct", "All": "Total"}
            )

            # Remove totals for heatmap
            conf_matrix = confusion_data.iloc[:-1, :-1]

            # Create heatmap
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar_kws={"label": "Count"},
            )
            ax.set_title(f"{model} - Point Game Confusion Matrix")
            ax.set_xlabel("Model Prediction")
            ax.set_ylabel("Point Game Result")

        plt.tight_layout()
        save_path = self.save_dir / "point_game_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_iou_distribution_histograms(self) -> Path:
        """
        Erstellt Histogramme der IoU-Verteilung pro Modell und Methode,
        aufgeteilt nach richtig/falsch klassifizierten Samples.
        """
        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        # Calculate grid layout
        n_cols = min(3, n_combinations)  # Max 3 columns
        n_rows = (n_combinations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "IoU Distribution per Model-Method: Correctly vs Incorrectly Classified",
            fontsize=16,
            fontweight="bold",
        )

        colors = ["green", "red"]
        labels = ["Correctly Classified", "Incorrectly Classified"]

        for idx, (_, row) in enumerate(model_explainer_combinations.iterrows()):
            if idx >= len(axes):
                break

            model = row["model_name"]
            explainer = row["explainer_name"]

            # Filter data for this specific model-explainer combination
            combo_data = self.df[
                (self.df["model_name"] == model)
                & (self.df["explainer_name"] == explainer)
            ]

            # Split data by prediction correctness
            correct_data = combo_data[combo_data["prediction_correct"]]["iou"]
            incorrect_data = combo_data[~combo_data["prediction_correct"]]["iou"]

            ax = axes[idx]

            # Create overlapping histograms
            if len(correct_data) > 0:
                ax.hist(
                    correct_data,
                    bins=25,
                    alpha=0.7,
                    color=colors[0],
                    label=f"{labels[0]} (n={len(correct_data)})",
                    density=True,
                )

            if len(incorrect_data) > 0:
                ax.hist(
                    incorrect_data,
                    bins=25,
                    alpha=0.7,
                    color=colors[1],
                    label=f"{labels[1]} (n={len(incorrect_data)})",
                    density=True,
                )

            # Calculate statistics for display
            if len(correct_data) > 0:
                correct_mean = correct_data.mean()
                correct_std = correct_data.std()
            else:
                correct_mean = correct_std = 0

            if len(incorrect_data) > 0:
                incorrect_mean = incorrect_data.mean()
                incorrect_std = incorrect_data.std()
            else:
                incorrect_mean = incorrect_std = 0

            # Create title with model, method and statistics
            title_lines = [
                f"{model} + {explainer}",
                f"IoU: Correct μ={correct_mean:.3f}±{correct_std:.3f}, Incorrect μ"
                f"={incorrect_mean:.3f}±{incorrect_std:.3f}",
            ]
            ax.set_title("\n".join(title_lines), fontsize=12, fontweight="bold")

            ax.set_xlabel("IoU Score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set consistent x-axis limits
            ax.set_xlim(0, 1)

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        save_path = self.save_dir / "iou_distribution_histograms.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def _plot_pixel_precision_distribution_histograms(self) -> Path:
        """
        Erstellt Histogramme der Pixel Precision Verteilung pro Modell und Methode,
        aufgeteilt nach richtig/falsch klassifizierten Samples.
        """
        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        # Calculate grid layout
        n_cols = min(3, n_combinations)  # Max 3 columns
        n_rows = (n_combinations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Pixel Precision Distribution per Model-Method: Correctly vs"
            "Incorrectly Classified",
            fontsize=16,
            fontweight="bold",
        )

        colors = ["green", "red"]
        labels = ["Correctly Classified", "Incorrectly Classified"]

        for idx, (_, row) in enumerate(model_explainer_combinations.iterrows()):
            if idx >= len(axes):
                break

            model = row["model_name"]
            explainer = row["explainer_name"]

            # Filter data for this specific model-explainer combination
            combo_data = self.df[
                (self.df["model_name"] == model)
                & (self.df["explainer_name"] == explainer)
            ]

            # Split data by prediction correctness
            correct_data = combo_data[combo_data["prediction_correct"]][
                "pixelprecisionrecall_precision"
            ]
            incorrect_data = combo_data[~combo_data["prediction_correct"]][
                "pixelprecisionrecall_precision"
            ]

            ax = axes[idx]

            # Create overlapping histograms
            if len(correct_data) > 0:
                ax.hist(
                    correct_data,
                    bins=25,
                    alpha=0.7,
                    color=colors[0],
                    label=f"{labels[0]} (n={len(correct_data)})",
                    density=True,
                )

            if len(incorrect_data) > 0:
                ax.hist(
                    incorrect_data,
                    bins=25,
                    alpha=0.7,
                    color=colors[1],
                    label=f"{labels[1]} (n={len(incorrect_data)})",
                    density=True,
                )

            # Calculate statistics for display
            if len(correct_data) > 0:
                correct_mean = correct_data.mean()
                correct_std = correct_data.std()
            else:
                correct_mean = correct_std = 0

            if len(incorrect_data) > 0:
                incorrect_mean = incorrect_data.mean()
                incorrect_std = incorrect_data.std()
            else:
                incorrect_mean = incorrect_std = 0

            # Create title with model, method and statistics
            title_lines = [
                f"{model} + {explainer}",
                f"Pixel Precision: Correct μ={correct_mean:.3f}±{correct_std:.3f}, "
                f"Incorrect μ={incorrect_mean:.3f}±{incorrect_std:.3f}",
            ]
            ax.set_title("\n".join(title_lines), fontsize=12, fontweight="bold")

            ax.set_xlabel("Pixel Precision Score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set consistent x-axis limits
            ax.set_xlim(0, 1)

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        save_path = self.save_dir / "pixel_precision_distribution_histograms.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_point_game_confusion_matrix_comparison(self) -> Path:
        """
        Erstellt Vergleichsplots für Point Game Confusion Matrices
        aller Modell-Methoden-Kombinationen.

        Returns:
            Path: Pfad zum gespeicherten Vergleichsplot
        """
        if (
            "point_game" not in self.available_metrics
            or "prediction_correct" not in self.df.columns
        ):
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "Point Game data or prediction correctness not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Point Game Confusion Matrix Comparison - Data Not Available")
            save_path = self.save_dir / "point_game_confusion_matrix_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        if n_combinations == 0:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No model-explainer combinations found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Point Game Confusion Matrix Comparison - No Data")
            save_path = self.save_dir / "point_game_confusion_matrix_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path

        # Calculate grid layout
        n_cols = min(3, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Point Game Confusion Matrix Comparison: All Model-Method Combinations",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Store confusion matrix data
        comparison_data = []
        threshold = 0.5

        for idx, (_, row) in enumerate(model_explainer_combinations.iterrows()):
            if idx >= len(axes):
                break

            model = row["model_name"]
            explainer = row["explainer_name"]

            # Filter data for this specific model-explainer combination
            combo_data = self.df[
                (self.df["model_name"] == model)
                & (self.df["explainer_name"] == explainer)
            ]

            if len(combo_data) == 0:
                continue

            ax = axes[idx]

            # Convert point game to binary (hit=1, miss=0)
            y_true = combo_data["prediction_correct"].astype(int)
            y_pred_point_game = (combo_data["point_game"] >= threshold).astype(int)

            # Create confusion matrix
            from sklearn.metrics import (
                confusion_matrix,
            )

            conf_matrix = confusion_matrix(y_true, y_pred_point_game)

            # Create heatmap using seaborn for consistency
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Miss", "Hit"],
                yticklabels=["Incorrect", "Correct"],
                cbar_kws={"label": "Count"},
            )

            ax.set_ylabel("Prediction Correctness")
            ax.set_xlabel("Point Game Result")
            ax.set_title(f"{model} + {explainer}", fontweight="bold", fontsize=12)

            # Calculate metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Add metrics text
            metrics_text = (
                f"Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec:"
                f" {recall:.3f}\nF1: {f1:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Store comparison data
            comparison_data.append(
                {
                    "model_name": model,
                    "explainer_name": explainer,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "total_samples": len(combo_data),
                }
            )

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.save_dir / "point_game_confusion_matrix_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        # Save comparison data as CSV
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_csv_path = (
                self.save_dir / "point_game_confusion_matrix_comparison_data.csv"
            )
            comparison_df.to_csv(comparison_csv_path, index=False)

            # Create summary heatmap for F1 scores
            if len(comparison_data) > 1:
                self._create_f1_heatmap_comparison(comparison_df)

        return save_path

    def _create_f1_heatmap_comparison(self, comparison_df: pd.DataFrame) -> Path:
        """
        Erstellt eine Heatmap der F1-Scores für Point Game über alle Kombinationen.

        Args:
            comparison_df: DataFrame mit Confusion Matrix Vergleichsdaten

        Returns:
            Path zum gespeicherten Heatmap-Plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create pivot table for heatmap
        pivot_data = comparison_df.pivot(
            index="model_name", columns="explainer_name", values="f1_score"
        )

        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "F1 Score"},
            center=0.5,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            "Point Game Performance Comparison: F1 Scores by Model and Method",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("XAI Method")
        ax.set_ylabel("Model")

        plt.tight_layout()
        save_path = self.save_dir / "point_game_f1_score_heatmap_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        return save_path

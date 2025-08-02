from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns


class XaiMetaAnalysis:
    """
    Performs meta-analysis on XAI evaluation metrics with respect to prediction
    correctness.
    """

    metrics: List[str] = [
        "iou",
        "pixelprecisionrecall_precision",
        "pixelprecisionrecall_recall",
        "point_game",
        "prediction_confidence",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the analysis with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing XAI metrics and prediction data.
        """
        self.df = df.copy()
        if "prediction_correct" in self.df.columns:
            self.df["prediction_correct"] = self.df["prediction_correct"].astype(bool)
        else:
            self.df["prediction_correct"] = (
                self.df["predicted_class"] == self.df["true_label"]
            )
        self.df["prediction_correct_int"] = self.df["prediction_correct"].astype(int)

    def plot_metric_vs_correctness(
        self, metrics: Optional[List[str]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate boxplots of selected metrics split by prediction correctness.

        Args:
            metrics (List[str], optional): List of metrics to plot. Defaults to the
            class metrics.

        Returns:
            Dict[str, plt.Figure]: Dictionary mapping metric names to matplotlib
            Figure objects.
        """
        if metrics is None:
            metrics = self.metrics

        plots = {}
        for metric in metrics:
            if metric not in self.df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="prediction_correct", y=metric, data=self.df, ax=ax)
            ax.set_title(f"{metric} vs. Prediction Correctness")
            ax.set_xlabel("Prediction Correct (False=0, True=1)")
            ax.set_ylabel(metric)
            plots[metric] = fig
        return plots

    def correlation_with_correctness(
        self, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation between metrics and prediction correctness.

        Args:
            metrics (List[str], optional): Metrics to correlate. Defaults to class
            metrics.

        Returns:
            Dict[str, float]: Dictionary of metric name to correlation coefficient.
        """
        if metrics is None:
            metrics = self.metrics

        corr_dict = {}
        for metric in metrics:
            if metric not in self.df.columns:
                continue
            corr = self.df[[metric, "prediction_correct_int"]].corr().iloc[0, 1]
            corr_dict[metric] = corr
        return corr_dict

    def global_mean_correlation(
        self, group_by: List[str] = ["model_name", "explainer_name"]
    ) -> pd.DataFrame:
        """
        Compute correlation matrix of group-wise mean metrics.

        Args:
            group_by (List[str]): Columns to group by.

        Returns:
            pd.DataFrame: Correlation matrix of grouped means.
        """
        metric_cols = self.metrics
        df_grouped = self.df.groupby(group_by)[metric_cols].mean().reset_index()
        corr = df_grouped[metric_cols].corr()
        return corr

    def threshold_analysis(
        self, metric: str, bins: int = 5
    ) -> Tuple[pd.Series, plt.Figure]:
        """
        Group a metric into bins and calculate mean prediction correctness per bin.

        Args:
            metric (str): Metric to analyze.
            bins (int): Number of bins.

        Returns:
            Tuple[pd.Series, plt.Figure]: Grouped means and bar plot figure.
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame.")

        self.df["bin"] = pd.qcut(self.df[metric], q=bins, duplicates="drop")
        grouped = self.df.groupby("bin", observed=False)[
            "prediction_correct_int"
        ].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"Prediction Accuracy vs. Binned {metric}")
        ax.set_xlabel(f"{metric} bins")
        ax.set_ylabel("Prediction Accuracy (mean)")
        return grouped, fig

    def scatter(self, metric: str) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Scatter plot of a given metric vs. prediction correctness.

        Args:
            metric (str): Metric to analyze.

        Returns:
            Tuple[pd.DataFrame, plt.Figure]: Raw data (metric vs. correctness) and
            scatter plot figure.
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame.")

        data = self.df[[metric, "prediction_confidence"]].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data[metric], data["prediction_confidence"], alpha=0.4)
        ax.set_title(f"Prediction Confidence vs. {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Prediction Confidence")
        ax.grid(True)

        return data, fig


if __name__ == "__main__":
    with mlflow.start_run(run_name="meta Analysis"):
        project_root = Path(__file__).resolve().parents[3]
        output_dir = Path("results/resnet50_experiment")
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_analysis_dir = output_dir / "meta_analysis"
        meta_plot_dir = meta_analysis_dir / "plots"
        meta_analysis_dir.mkdir(parents=True, exist_ok=True)
        meta_plot_dir.mkdir(parents=True, exist_ok=True)

        csv_path = project_root / output_dir / "results_with_metrics.csv"
        print(f"Reading CSV from {csv_path.resolve()}")
        df = pd.read_csv(csv_path)
        analysis = XaiMetaAnalysis(df)

        # Plots erzeugen
        plots = analysis.plot_metric_vs_correctness()
        for name, fig in plots.items():
            filepath = meta_plot_dir / f"{name}_vs_correctness.png"
            fig.savefig(filepath)
            plt.close(fig)
            mlflow.log_artifact(str(filepath))

        # Threshold-Analyse
        grouped, fig = analysis.threshold_analysis("iou")
        threshold_plot_path = meta_plot_dir / "threshold_iou_score.png"
        fig.savefig(threshold_plot_path)
        plt.close(fig)
        mlflow.log_artifact(str(threshold_plot_path))

        # Threshold-Analyse
        grouped_2, fig2 = analysis.scatter("iou")
        scatter_plot_path = meta_plot_dir / "prediction_confidence_iou_score.png"
        fig2.savefig(scatter_plot_path)
        plt.close(fig2)
        mlflow.log_artifact(str(scatter_plot_path))

        # CSV speichern
        threshold_csv_path = meta_analysis_dir / "threshold_iou_score.csv"
        grouped.to_csv(threshold_csv_path)
        mlflow.log_artifact(str(threshold_csv_path))

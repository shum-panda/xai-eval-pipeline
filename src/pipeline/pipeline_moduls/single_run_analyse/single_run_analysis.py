from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class SingleRunAnalyse:
    """
    Performs meta-analysis on XAI evaluation metrics with respect to prediction
    correctness.
    """

    metrics: List[str] = [
        "iou",
        "pixel_precision",
        "pixel_recall",
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
        self, metrics: Optional[List[str]] = None, save_data: bool = True, save_dir: Optional[Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate boxplots of selected metrics split by prediction correctness.

        Args:
            metrics (List[str], optional): List of metrics to plot. Defaults to the
            class metrics.
            save_data (bool): Whether to save underlying data alongside plots
            save_dir (Optional[Path]): Directory to save data files

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
            
            # Save underlying data if requested
            if save_data and save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save boxplot data summary
                boxplot_data = self.df.groupby('prediction_correct')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median',
                    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
                ]).round(6)
                boxplot_data.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
                
                data_file = save_dir / f"{metric}_vs_correctness_data.csv"
                boxplot_data.to_csv(data_file)
                
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

    def calculate_model_method_f1_scores(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Berechnet F1-Scores für Point Game pro Modell und Methode.

        Args:
            threshold: Schwellenwert für Point Game Hit/Miss

        Returns:
            DataFrame mit detaillierten F1-Score Metriken
        """
        if "point_game" not in self.df.columns:
            return pd.DataFrame()

        results = []

        # Group by model and explainer if available, otherwise just by available columns
        groupby_cols = []
        if "model_name" in self.df.columns:
            groupby_cols.append("model_name")
        if "explainer_name" in self.df.columns:
            groupby_cols.append("explainer_name")

        if not groupby_cols:
            # Fallback: analyze the entire dataset as one group
            groupby_cols = ["dataset"]  # Use a dummy column
            self.df["dataset"] = "all"

        for group_values, group_data in self.df.groupby(groupby_cols):
            if isinstance(group_values, str):
                group_values = [group_values]

            # Create group identifier
            group_dict = dict(zip(groupby_cols, group_values))

            # Convert point game to binary
            y_true = group_data["prediction_correct"].astype(int)
            y_pred_point_game = (group_data["point_game"] >= threshold).astype(int)

            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_point_game)) > 1:
                f1 = f1_score(y_true, y_pred_point_game)
                precision = precision_score(y_true, y_pred_point_game)
                recall = recall_score(y_true, y_pred_point_game)

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_point_game).ravel()

                result = {
                    **group_dict,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": tp,
                    "true_negatives": tn,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "total_samples": len(group_data),
                    "threshold_used": threshold,
                }
            else:
                result = {
                    **group_dict,
                    "f1_score": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "true_positives": np.nan,
                    "true_negatives": np.nan,
                    "false_positives": np.nan,
                    "false_negatives": np.nan,
                    "total_samples": len(group_data),
                    "threshold_used": threshold,
                }

            results.append(result)

        return pd.DataFrame(results)

    def calculate_statistical_tests_for_all_metrics(self) -> pd.DataFrame:
        """
        Berechnet statistische Tests (Mann-Whitney U, Cohen's d, KS-Test) für alle Metriken
        zwischen richtig und falsch klassifizierten Samples.

        Returns:
            DataFrame mit statistischen Test-Ergebnissen für jede Metrik
        """
        from scipy import stats
        
        # Define available metrics (check for different column naming patterns)
        metrics_map = {
            "iou": ["iou"],
            "pixel_precision": ["pixel_precision", "pixelprecisionrecall_precision"], 
            "pixel_recall": ["pixel_recall", "pixelprecisionrecall_recall"],
            "point_game": ["point_game"],
            "prediction_confidence": ["prediction_confidence"]
        }
        
        # Find actual column names in the dataframe
        available_metrics = []
        metric_name_map = {}  # Maps display name to actual column name
        
        for display_name, possible_cols in metrics_map.items():
            for col in possible_cols:
                if col in self.df.columns:
                    available_metrics.append(display_name)
                    metric_name_map[display_name] = col
                    break
        
        if not available_metrics:
            return pd.DataFrame()
            
        results = []
        
        for metric in available_metrics:
            actual_col = metric_name_map[metric]
            
            # Split data by prediction correctness
            correct_data = self.df[self.df["prediction_correct"] == True][actual_col].dropna()
            incorrect_data = self.df[self.df["prediction_correct"] == False][actual_col].dropna()
            
            if len(correct_data) > 0 and len(incorrect_data) > 0:
                # Mann-Whitney U test (non-parametric)
                mw_statistic, mw_p_value = stats.mannwhitneyu(correct_data, incorrect_data, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(correct_data) - 1) * correct_data.std()**2 + 
                                    (len(incorrect_data) - 1) * incorrect_data.std()**2) / 
                                   (len(correct_data) + len(incorrect_data) - 2))
                cohens_d = (correct_data.mean() - incorrect_data.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Kolmogorov-Smirnov test for distribution differences
                ks_statistic, ks_p_value = stats.ks_2samp(correct_data, incorrect_data)
                
                # Welch's t-test (unequal variances)
                welch_statistic, welch_p_value = stats.ttest_ind(correct_data, incorrect_data, equal_var=False)
                
                # Levene test for equal variances
                levene_statistic, levene_p_value = stats.levene(correct_data, incorrect_data)
                
                result = {
                    'metric': metric,
                    'actual_column': actual_col,
                    'correct_samples': len(correct_data),
                    'incorrect_samples': len(incorrect_data),
                    'correct_mean': correct_data.mean(),
                    'incorrect_mean': incorrect_data.mean(),
                    'correct_std': correct_data.std(),
                    'incorrect_std': incorrect_data.std(),
                    'mean_difference': correct_data.mean() - incorrect_data.mean(),
                    'mann_whitney_u_statistic': mw_statistic,
                    'mann_whitney_u_p_value': mw_p_value,
                    'mann_whitney_u_significant': mw_p_value < 0.05,
                    'cohens_d': cohens_d,
                    'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'ks_significant': ks_p_value < 0.05,
                    'welch_t_statistic': welch_statistic,
                    'welch_t_p_value': welch_p_value,
                    'welch_t_significant': welch_p_value < 0.05,
                    'levene_statistic': levene_statistic,
                    'levene_p_value': levene_p_value,
                    'equal_variances': levene_p_value > 0.05
                }
            else:
                result = {
                    'metric': metric,
                    'actual_column': actual_col,
                    'correct_samples': len(correct_data),
                    'incorrect_samples': len(incorrect_data),
                    'correct_mean': correct_data.mean() if len(correct_data) > 0 else np.nan,
                    'incorrect_mean': incorrect_data.mean() if len(incorrect_data) > 0 else np.nan,
                    'correct_std': correct_data.std() if len(correct_data) > 0 else np.nan,
                    'incorrect_std': incorrect_data.std() if len(incorrect_data) > 0 else np.nan,
                    'mean_difference': np.nan,
                    'mann_whitney_u_statistic': np.nan,
                    'mann_whitney_u_p_value': np.nan,
                    'mann_whitney_u_significant': False,
                    'cohens_d': np.nan,
                    'cohens_d_interpretation': 'insufficient_data',
                    'ks_statistic': np.nan,
                    'ks_p_value': np.nan,
                    'ks_significant': False,
                    'welch_t_statistic': np.nan,
                    'welch_t_p_value': np.nan,
                    'welch_t_significant': False,
                    'levene_statistic': np.nan,
                    'levene_p_value': np.nan,
                    'equal_variances': np.nan
                }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpretiert Cohen's d Effektgröße."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def calculate_iou_distribution_stats(self) -> pd.DataFrame:
        """
        Berechnet IoU-Verteilungsstatistiken pro Modell/Methode und Korrektheit.

        Returns:
            DataFrame mit IoU-Statistiken aufgeteilt nach Prediction Correctness
        """
        if "iou" not in self.df.columns:
            return pd.DataFrame()

        results = []

        # Group by model and explainer if available
        groupby_cols = []
        if "model_name" in self.df.columns:
            groupby_cols.append("model_name")
        if "explainer_name" in self.df.columns:
            groupby_cols.append("explainer_name")

        if not groupby_cols:
            groupby_cols = ["dataset"]
            self.df["dataset"] = "all"

        for group_values, group_data in self.df.groupby(groupby_cols):
            if isinstance(group_values, str):
                group_values = [group_values]

            group_dict = dict(zip(groupby_cols, group_values))

            # Split by prediction correctness
            correct_data = group_data[group_data["prediction_correct"] == True]["iou"]
            incorrect_data = group_data[group_data["prediction_correct"] == False][
                "iou"
            ]

            # Calculate statistics for correctly classified
            if len(correct_data) > 0:
                correct_stats = {
                    **group_dict,
                    "prediction_correctness": "correct",
                    "count": len(correct_data),
                    "mean_iou": correct_data.mean(),
                    "std_iou": correct_data.std(),
                    "median_iou": correct_data.median(),
                    "min_iou": correct_data.min(),
                    "max_iou": correct_data.max(),
                    "q25_iou": correct_data.quantile(0.25),
                    "q75_iou": correct_data.quantile(0.75),
                }
                results.append(correct_stats)

            # Calculate statistics for incorrectly classified
            if len(incorrect_data) > 0:
                incorrect_stats = {
                    **group_dict,
                    "prediction_correctness": "incorrect",
                    "count": len(incorrect_data),
                    "mean_iou": incorrect_data.mean(),
                    "std_iou": incorrect_data.std(),
                    "median_iou": incorrect_data.median(),
                    "min_iou": incorrect_data.min(),
                    "max_iou": incorrect_data.max(),
                    "q25_iou": incorrect_data.quantile(0.25),
                    "q75_iou": incorrect_data.quantile(0.75),
                }
                results.append(incorrect_stats)

        return pd.DataFrame(results)

    def plot_iou_histograms_by_correctness(self, save_dir: Path) -> Dict[str, Path]:
        """
        Erstellt verbesserte Histogramme der IoU-Verteilung pro Modell und Methode,
        aufgeteilt nach richtig/falsch klassifizierten Samples.
        
        Args:
            save_dir: Verzeichnis zum Speichern der Plots
            
        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        if "iou" not in self.df.columns:
            return {}
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        if n_combinations == 0:
            return plot_paths

        # Calculate grid layout - improved layout for better readability
        n_cols = min(2, n_combinations)  # Max 2 columns for better readability
        n_rows = (n_combinations + n_cols - 1) // n_cols

        # Improved figure size and style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "IoU Distribution Analysis: Model Performance by Prediction Correctness",
            fontsize=20,
            fontweight="bold",
            y=0.98
        )

        # Improved colors with better contrast
        colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson
        labels = ["Correctly Classified", "Incorrectly Classified"]

        # Save histogram data for each combination
        histogram_data = []

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

            # Create improved overlapping histograms
            bins = np.linspace(0, 1, 31)  # Fixed bins for consistency
            
            if len(correct_data) > 0:
                counts_correct, bin_edges, _ = ax.hist(
                    correct_data,
                    bins=bins,
                    alpha=0.6,
                    color=colors[0],
                    label=f"{labels[0]} (n={len(correct_data)})",
                    density=True,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Add mean line for correct predictions
                correct_mean = correct_data.mean()
                ax.axvline(correct_mean, color=colors[0], linestyle='--', linewidth=2, alpha=0.8)
                ax.text(correct_mean + 0.02, ax.get_ylim()[1] * 0.8, 
                       f'μ={correct_mean:.3f}', rotation=90, color=colors[0], fontweight='bold')
            else:
                counts_correct, bin_edges = np.array([]), np.array([])

            if len(incorrect_data) > 0:
                counts_incorrect, bin_edges_inc, _ = ax.hist(
                    incorrect_data,
                    bins=bins,
                    alpha=0.6,
                    color=colors[1],
                    label=f"{labels[1]} (n={len(incorrect_data)})",
                    density=True,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Add mean line for incorrect predictions
                incorrect_mean = incorrect_data.mean()
                ax.axvline(incorrect_mean, color=colors[1], linestyle='--', linewidth=2, alpha=0.8)
                ax.text(incorrect_mean + 0.02, ax.get_ylim()[1] * 0.9, 
                       f'μ={incorrect_mean:.3f}', rotation=90, color=colors[1], fontweight='bold')
            else:
                counts_incorrect = np.array([])

            # Save histogram data
            if len(correct_data) > 0 or len(incorrect_data) > 0:
                combo_hist_data = {
                    "model_name": model,
                    "explainer_name": explainer,
                    "correct_samples": len(correct_data),
                    "incorrect_samples": len(incorrect_data),
                    "correct_iou_mean": correct_data.mean() if len(correct_data) > 0 else np.nan,
                    "correct_iou_std": correct_data.std() if len(correct_data) > 0 else np.nan,
                    "correct_iou_median": correct_data.median() if len(correct_data) > 0 else np.nan,
                    "incorrect_iou_mean": incorrect_data.mean() if len(incorrect_data) > 0 else np.nan,
                    "incorrect_iou_std": incorrect_data.std() if len(incorrect_data) > 0 else np.nan,
                    "incorrect_iou_median": incorrect_data.median() if len(incorrect_data) > 0 else np.nan,
                }
                histogram_data.append(combo_hist_data)

            # Calculate statistics for display
            if len(correct_data) > 0:
                correct_mean = correct_data.mean()
                correct_std = correct_data.std()
                correct_median = correct_data.median()
            else:
                correct_mean = correct_std = correct_median = 0

            if len(incorrect_data) > 0:
                incorrect_mean = incorrect_data.mean()
                incorrect_std = incorrect_data.std()
                incorrect_median = incorrect_data.median()
            else:
                incorrect_mean = incorrect_std = incorrect_median = 0

            # Create improved title with better formatting
            title_main = f"{model.upper()} + {explainer.replace('_', ' ').title()}"
            stats_line1 = f"Correct: μ={correct_mean:.3f} σ={correct_std:.3f} med={correct_median:.3f}"
            stats_line2 = f"Incorrect: μ={incorrect_mean:.3f} σ={incorrect_std:.3f} med={incorrect_median:.3f}"
            
            ax.text(0.02, 0.98, title_main, transform=ax.transAxes, fontsize=14, 
                   fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            ax.text(0.02, 0.88, stats_line1, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color=colors[0], fontweight='bold')
            ax.text(0.02, 0.83, stats_line2, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color=colors[1], fontweight='bold')

            # Improved styling
            ax.set_xlabel("IoU Score", fontsize=12, fontweight='bold')
            ax.set_ylabel("Density", fontsize=12, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlim(0, 1)
            
            # Add subtle background color
            ax.set_facecolor('#f8f8f8')
            
            # Improve tick formatting
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Hide unused subplots with style
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        plot_path = save_dir / "iou_distribution_histograms_improved.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        plot_paths["iou_histograms"] = plot_path

        # Save histogram statistics data
        if histogram_data:
            hist_stats_df = pd.DataFrame(histogram_data)
            hist_stats_path = save_dir / "iou_histogram_statistics.csv"
            hist_stats_df.to_csv(hist_stats_path, index=False)
            plot_paths["iou_histogram_data"] = hist_stats_path

        return plot_paths

    def plot_prediction_correctness_histograms(self, save_dir: Path) -> Dict[str, Path]:
        """
        Erstellt verbesserte Histogramme für alle Metriken aufgeteilt nach prediction correctness.
        
        Args:
            save_dir: Verzeichnis zum Speichern der Plots
            
        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        # Metrics to analyze (excluding iou which has its own method)
        metrics_to_plot = [m for m in self.metrics if m in self.df.columns and m != "iou"]
        
        if not metrics_to_plot:
            return plot_paths

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        
        # Improved colors with better contrast
        colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson
        labels = ["Correctly Classified", "Incorrectly Classified"]
        
        # Create plots for each metric
        for metric in metrics_to_plot:
            n_combinations = len(model_explainer_combinations)
            
            if n_combinations == 0:
                continue

            # Calculate grid layout - improved for better readability
            n_cols = min(2, n_combinations)  # Max 2 columns for better readability
            n_rows = (n_combinations + n_cols - 1) // n_cols

            # Improved figure size and style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))

            # Handle single subplot case
            if n_combinations == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            # Get display name for metric
            display_name = metric.replace("_", " ").title()
            if metric == "pixelprecisionrecall_precision":
                display_name = "Pixel Precision"
            elif metric == "pixelprecisionrecall_recall":
                display_name = "Pixel Recall"
            elif metric == "point_game":
                display_name = "Point Game"

            fig.suptitle(
                f"{display_name} Distribution Analysis: Model Performance by Prediction Correctness",
                fontsize=20,
                fontweight="bold",
                y=0.98
            )

            # Save histogram data for each combination
            histogram_data = []

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
                correct_data = combo_data[combo_data["prediction_correct"]][metric]
                incorrect_data = combo_data[~combo_data["prediction_correct"]][metric]

                ax = axes[idx]

                # Create improved overlapping histograms
                # Determine appropriate bins based on metric
                if metric in ["pixelprecisionrecall_precision", "pixelprecisionrecall_recall", "point_game"]:
                    bins = np.linspace(0, 1, 31)  # Fixed bins for 0-1 range metrics
                else:
                    bins = 25  # Adaptive bins for other metrics
                    
                if len(correct_data) > 0:
                    ax.hist(
                        correct_data,
                        bins=bins,
                        alpha=0.6,
                        color=colors[0],
                        label=f"{labels[0]} (n={len(correct_data)})",
                        density=True,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Add mean line for correct predictions
                    correct_mean = correct_data.mean()
                    ax.axvline(correct_mean, color=colors[0], linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Position mean label smartly
                    y_pos = ax.get_ylim()[1] * 0.8
                    x_offset = 0.02 if correct_mean < 0.8 else -0.05
                    ax.text(correct_mean + x_offset, y_pos, 
                           f'μ={correct_mean:.3f}', rotation=90, color=colors[0], fontweight='bold')

                if len(incorrect_data) > 0:
                    ax.hist(
                        incorrect_data,
                        bins=bins,
                        alpha=0.6,
                        color=colors[1],
                        label=f"{labels[1]} (n={len(incorrect_data)})",
                        density=True,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Add mean line for incorrect predictions
                    incorrect_mean = incorrect_data.mean()
                    ax.axvline(incorrect_mean, color=colors[1], linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Position mean label smartly
                    y_pos = ax.get_ylim()[1] * 0.9
                    x_offset = 0.02 if incorrect_mean < 0.8 else -0.05
                    ax.text(incorrect_mean + x_offset, y_pos, 
                           f'μ={incorrect_mean:.3f}', rotation=90, color=colors[1], fontweight='bold')

                # Save histogram statistics
                combo_hist_data = {
                    "model_name": model,
                    "explainer_name": explainer,
                    "metric": metric,
                    "correct_samples": len(correct_data),
                    "incorrect_samples": len(incorrect_data),
                    "correct_mean": correct_data.mean() if len(correct_data) > 0 else np.nan,
                    "correct_std": correct_data.std() if len(correct_data) > 0 else np.nan,
                    "correct_median": correct_data.median() if len(correct_data) > 0 else np.nan,
                    "incorrect_mean": incorrect_data.mean() if len(incorrect_data) > 0 else np.nan,
                    "incorrect_std": incorrect_data.std() if len(incorrect_data) > 0 else np.nan,
                    "incorrect_median": incorrect_data.median() if len(incorrect_data) > 0 else np.nan,
                }
                histogram_data.append(combo_hist_data)

                # Calculate statistics for display
                if len(correct_data) > 0:
                    correct_mean = correct_data.mean()
                    correct_std = correct_data.std()
                    correct_median = correct_data.median()
                else:
                    correct_mean = correct_std = correct_median = 0

                if len(incorrect_data) > 0:
                    incorrect_mean = incorrect_data.mean()
                    incorrect_std = incorrect_data.std()
                    incorrect_median = incorrect_data.median()
                else:
                    incorrect_mean = incorrect_std = incorrect_median = 0

                # Create improved title with better formatting
                title_main = f"{model.upper()} + {explainer.replace('_', ' ').title()}"
                stats_line1 = f"Correct: μ={correct_mean:.3f} σ={correct_std:.3f} med={correct_median:.3f}"
                stats_line2 = f"Incorrect: μ={incorrect_mean:.3f} σ={incorrect_std:.3f} med={incorrect_median:.3f}"
                
                ax.text(0.02, 0.98, title_main, transform=ax.transAxes, fontsize=14, 
                       fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                ax.text(0.02, 0.88, stats_line1, transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top', color=colors[0], fontweight='bold')
                ax.text(0.02, 0.83, stats_line2, transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top', color=colors[1], fontweight='bold')

                # Improved styling
                ax.set_xlabel(f"{display_name} Score", fontsize=12, fontweight='bold')
                ax.set_ylabel("Density", fontsize=12, fontweight='bold')
                ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

                # Set consistent x-axis limits based on metric type
                if metric in ["pixelprecisionrecall_precision", "pixelprecisionrecall_recall", "point_game"]:
                    ax.set_xlim(0, 1)
                
                # Add subtle background color
                ax.set_facecolor('#f8f8f8')
                
                # Improve tick formatting
                ax.tick_params(axis='both', which='major', labelsize=10)

            # Hide unused subplots with style
            for idx in range(n_combinations, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
            plot_path = save_dir / f"{metric}_distribution_histograms_improved.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()
            plot_paths[f"{metric}_histograms"] = plot_path

            # Save histogram statistics data
            if histogram_data:
                hist_stats_df = pd.DataFrame(histogram_data)
                hist_stats_path = save_dir / f"{metric}_histogram_statistics.csv"
                hist_stats_df.to_csv(hist_stats_path, index=False)
                plot_paths[f"{metric}_histogram_data"] = hist_stats_path

        return plot_paths

    def plot_pixel_precision_histograms_by_correctness(self, save_dir: Path) -> Dict[str, Path]:
        """
        Erstellt Histogramme für Pixel Precision aufgeteilt nach prediction correctness.
        
        Args:
            save_dir: Verzeichnis zum Speichern der Plots und Daten
            
        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        if "pixelprecisionrecall_precision" not in self.df.columns:
            return {}
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        if n_combinations == 0:
            return plot_paths

        # Calculate grid layout
        n_cols = min(2, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Pixel Precision Distribution: Performance by Prediction Correctness",
            fontsize=20,
            fontweight="bold",
            y=0.98
        )

        # Colors for correct/incorrect predictions
        colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson
        labels = ["Correctly Classified", "Incorrectly Classified"]
        
        histogram_data = []

        for idx, (_, combination) in enumerate(model_explainer_combinations.iterrows()):
            model_name = combination["model_name"]
            explainer_name = combination["explainer_name"]

            # Filter data for this combination
            combo_data = self.df[
                (self.df["model_name"] == model_name) & 
                (self.df["explainer_name"] == explainer_name)
            ]

            if len(combo_data) == 0:
                continue

            ax = axes[idx] if n_combinations > 1 else axes[0]

            # Plot histograms for correct and incorrect predictions
            for correct_val, color, label in zip([True, False], colors, labels):
                subset = combo_data[combo_data["prediction_correct"] == correct_val]
                if len(subset) > 0:
                    pixel_precision_values = subset["pixelprecisionrecall_precision"].values
                    
                    # Use consistent bins for all plots
                    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
                    
                    ax.hist(
                        pixel_precision_values,
                        bins=bins,
                        alpha=0.7,
                        color=color,
                        label=f"{label} (n={len(subset)})",
                        density=True,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Add mean line
                    mean_val = pixel_precision_values.mean()
                    ax.axvline(
                        mean_val,
                        color=color,
                        linestyle='--',
                        alpha=0.8,
                        linewidth=2,
                        label=f"Mean: {mean_val:.3f}"
                    )
                    
                    # Store histogram data
                    histogram_data.append({
                        'model_name': model_name,
                        'explainer_name': explainer_name,
                        'prediction_correct': correct_val,
                        'mean_pixel_precision': mean_val,
                        'std_pixel_precision': pixel_precision_values.std(),
                        'count': len(subset),
                        'min_pixel_precision': pixel_precision_values.min(),
                        'max_pixel_precision': pixel_precision_values.max(),
                        'median_pixel_precision': np.median(pixel_precision_values)
                    })

            ax.set_title(f"{model_name} + {explainer_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Pixel Precision", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plot_path = save_dir / "pixel_precision_histograms_by_correctness.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["pixel_precision_histograms"] = plot_path

        # Save histogram data as CSV
        if histogram_data:
            hist_stats_df = pd.DataFrame(histogram_data)
            hist_stats_path = save_dir / "pixel_precision_histogram_data.csv"
            hist_stats_df.to_csv(hist_stats_path, index=False)
            plot_paths["pixel_precision_histogram_data"] = hist_stats_path

        return plot_paths

    def plot_pixel_recall_histograms_by_correctness(self, save_dir: Path) -> Dict[str, Path]:
        """
        Erstellt Histogramme für Pixel Recall aufgeteilt nach prediction correctness.
        
        Args:
            save_dir: Verzeichnis zum Speichern der Plots und Daten
            
        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        if "pixelprecisionrecall_recall" not in self.df.columns:
            return {}
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        if n_combinations == 0:
            return plot_paths

        # Calculate grid layout
        n_cols = min(2, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Pixel Recall Distribution: Performance by Prediction Correctness",
            fontsize=20,
            fontweight="bold",
            y=0.98
        )

        # Colors for correct/incorrect predictions
        colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson
        labels = ["Correctly Classified", "Incorrectly Classified"]
        
        histogram_data = []

        for idx, (_, combination) in enumerate(model_explainer_combinations.iterrows()):
            model_name = combination["model_name"]
            explainer_name = combination["explainer_name"]

            # Filter data for this combination
            combo_data = self.df[
                (self.df["model_name"] == model_name) & 
                (self.df["explainer_name"] == explainer_name)
            ]

            if len(combo_data) == 0:
                continue

            ax = axes[idx] if n_combinations > 1 else axes[0]

            # Plot histograms for correct and incorrect predictions
            for correct_val, color, label in zip([True, False], colors, labels):
                subset = combo_data[combo_data["prediction_correct"] == correct_val]
                if len(subset) > 0:
                    pixel_recall_values = subset["pixelprecisionrecall_recall"].values
                    
                    # Use consistent bins for all plots
                    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
                    
                    ax.hist(
                        pixel_recall_values,
                        bins=bins,
                        alpha=0.7,
                        color=color,
                        label=f"{label} (n={len(subset)})",
                        density=True,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Add mean line
                    mean_val = pixel_recall_values.mean()
                    ax.axvline(
                        mean_val,
                        color=color,
                        linestyle='--',
                        alpha=0.8,
                        linewidth=2,
                        label=f"Mean: {mean_val:.3f}"
                    )
                    
                    # Store histogram data
                    histogram_data.append({
                        'model_name': model_name,
                        'explainer_name': explainer_name,
                        'prediction_correct': correct_val,
                        'mean_pixel_recall': mean_val,
                        'std_pixel_recall': pixel_recall_values.std(),
                        'count': len(subset),
                        'min_pixel_recall': pixel_recall_values.min(),
                        'max_pixel_recall': pixel_recall_values.max(),
                        'median_pixel_recall': np.median(pixel_recall_values)
                    })

            ax.set_title(f"{model_name} + {explainer_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Pixel Recall", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plot_path = save_dir / "pixel_recall_histograms_by_correctness.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["pixel_recall_histograms"] = plot_path

        # Save histogram data as CSV
        if histogram_data:
            hist_stats_df = pd.DataFrame(histogram_data)
            hist_stats_path = save_dir / "pixel_recall_histogram_data.csv"
            hist_stats_df.to_csv(hist_stats_path, index=False)
            plot_paths["pixel_recall_histogram_data"] = hist_stats_path

        return plot_paths

    def plot_point_game_confusion_matrices(self, save_dir: Path, threshold: float = 0.5) -> Dict[str, Path]:
        """
        Erstellt Point Game Confusion Matrices für jede Modell-Explainer Kombination.

        Args:
            save_dir: Verzeichnis zum Speichern der Plots
            threshold: Schwellenwert für Point Game Hit/Miss

        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        if "point_game" not in self.df.columns or "prediction_correct" not in self.df.columns:
            return {}

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        # Get unique combinations of model and explainer
        model_explainer_combinations = (
            self.df.groupby(["model_name", "explainer_name"]).size().reset_index()
        )
        n_combinations = len(model_explainer_combinations)

        if n_combinations == 0:
            return plot_paths

        # Calculate grid layout
        n_cols = min(2, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

        # Handle single subplot case
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Point Game Confusion Matrices: Hit/Miss vs Correct/Incorrect Prediction",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        # Store confusion matrix data for each combination
        confusion_data = []

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
            from sklearn.metrics import confusion_matrix
            conf_matrix = confusion_matrix(y_true, y_pred_point_game)
            
            # Create labels for confusion matrix
            labels = [['TN\n(Incorrect & Miss)', 'FP\n(Incorrect & Hit)'],
                      ['FN\n(Correct & Miss)', 'TP\n(Correct & Hit)']]

            # Create heatmap
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            # Add text annotations
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(j, i, f'{labels[i][j]}\n{conf_matrix[i, j]}',
                           ha="center", va="center",
                           color="white" if conf_matrix[i, j] > thresh else "black",
                           fontweight='bold', fontsize=10)

            ax.set_ylabel('Prediction Correctness')
            ax.set_xlabel('Point Game Result')
            ax.set_title(f'{model} + {explainer}', fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Miss', 'Hit'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Incorrect', 'Correct'])

            # Calculate and display metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Add metrics text box
            metrics_text = f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Store data for CSV export
            confusion_data.append({
                'model_name': model,
                'explainer_name': explainer,
                'threshold': threshold,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_samples': len(combo_data)
            })

        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = save_dir / "point_game_confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        plot_paths["point_game_confusion_matrices"] = plot_path

        # Save confusion matrix data as CSV
        if confusion_data:
            confusion_df = pd.DataFrame(confusion_data)
            confusion_csv_path = save_dir / "point_game_confusion_matrices_data.csv"
            confusion_df.to_csv(confusion_csv_path, index=False)
            plot_paths["point_game_confusion_matrices_data"] = confusion_csv_path

        return plot_paths

    def generate_model_method_comparison_plots(self, save_dir: Path) -> Dict[str, Path]:
        """
        Erstellt Vergleichsplots für Modell/Methoden-Kombinationen.

        Args:
            save_dir: Verzeichnis zum Speichern der Plots

        Returns:
            Dictionary mit Plot-Namen und Pfaden
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = {}

        # 0. NEW: IoU Histogram Analysis
        print("Creating IoU distribution histograms...")
        iou_histogram_paths = self.plot_iou_histograms_by_correctness(save_dir)
        plot_paths.update(iou_histogram_paths)
        
        # 0.1 NEW: Prediction Correctness Histogram Analysis for other metrics
        print("Creating prediction correctness histograms...")
        pred_histogram_paths = self.plot_prediction_correctness_histograms(save_dir)
        plot_paths.update(pred_histogram_paths)

        # 0.2 NEW: Point Game Confusion Matrices
        print("Creating point game confusion matrices...")
        confusion_matrix_paths = self.plot_point_game_confusion_matrices(save_dir)
        plot_paths.update(confusion_matrix_paths)

        # 1. Enhanced F1 Score Comparison with Method Focus
        f1_scores = self.calculate_model_method_f1_scores()
        if not f1_scores.empty:
            # Create separate plots for models with multiple methods
            if (
                "model_name" in f1_scores.columns
                and "explainer_name" in f1_scores.columns
            ):
                models_with_multiple_methods = f1_scores.groupby("model_name")[
                    "explainer_name"
                ].nunique()
                models_with_multiple_methods = models_with_multiple_methods[
                    models_with_multiple_methods > 1
                ].index

                if len(models_with_multiple_methods) > 0:
                    # Create method comparison plot for models with multiple methods
                    fig, axes = plt.subplots(
                        1,
                        len(models_with_multiple_methods),
                        figsize=(8 * len(models_with_multiple_methods), 6),
                    )

                    if len(models_with_multiple_methods) == 1:
                        axes = [axes]

                    fig.suptitle(
                        "F1 Score Comparison: Methods per Model",
                        fontsize=16,
                        fontweight="bold",
                    )

                    for idx, model in enumerate(models_with_multiple_methods):
                        model_data = f1_scores[f1_scores["model_name"] == model]

                        ax = axes[idx]
                        bars = ax.bar(
                            range(len(model_data)),
                            model_data["f1_score"],
                            color=plt.cm.viridis(np.linspace(0, 1, len(model_data))),
                        )

                        # Add value labels on bars
                        for i, (bar, f1_val) in enumerate(
                            zip(bars, model_data["f1_score"])
                        ):
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.01,
                                f"{f1_val:.3f}",
                                ha="center",
                                va="bottom",
                                fontweight="bold",
                            )

                        ax.set_title(f"{model}\nMethod Comparison", fontweight="bold")
                        ax.set_ylabel("F1 Score")
                        ax.set_xlabel("XAI Method")
                        ax.set_xticks(range(len(model_data)))
                        ax.set_xticklabels(model_data["explainer_name"], rotation=45)
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(0, max(model_data["f1_score"]) * 1.1)

                    plt.tight_layout()
                    plot_path = save_dir / "f1_score_method_comparison.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    plot_paths["f1_score_method_comparison"] = plot_path

                # General heatmap for all combinations
                fig, ax = plt.subplots(figsize=(12, 8))
                pivot_data = f1_scores.pivot(
                    index="model_name", columns="explainer_name", values="f1_score"
                )

                mask = pivot_data.isnull()
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                    ax=ax,
                    mask=mask,
                    cbar_kws={"label": "F1 Score"},
                )
                ax.set_title(
                    "F1 Scores: Point Game vs Prediction Correctness\n(All Model-Method Combinations)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel("XAI Method")
                ax.set_ylabel("Model")

                plt.tight_layout()
                plot_path = save_dir / "f1_score_heatmap.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths["f1_score_heatmap"] = plot_path

        # 2. Enhanced IoU Distribution Comparison with Method-specific Analysis
        iou_stats = self.calculate_iou_distribution_stats()
        if not iou_stats.empty:
            # Create comprehensive IoU comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                "IoU Analysis: Model-Method Performance Comparison",
                fontsize=16,
                fontweight="bold",
            )

            # Top-left: Mean IoU comparison
            ax = axes[0, 0]
            correct_stats = iou_stats[iou_stats["prediction_correctness"] == "correct"]
            incorrect_stats = iou_stats[
                iou_stats["prediction_correctness"] == "incorrect"
            ]

            if not correct_stats.empty and not incorrect_stats.empty:
                x = np.arange(len(correct_stats))
                width = 0.35

                bars1 = ax.bar(
                    x - width / 2,
                    correct_stats["mean_iou"],
                    width,
                    label="Correctly Classified",
                    alpha=0.8,
                    color="green",
                )
                bars2 = ax.bar(
                    x + width / 2,
                    incorrect_stats["mean_iou"],
                    width,
                    label="Incorrectly Classified",
                    alpha=0.8,
                    color="red",
                )

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            f"{height:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                ax.set_xlabel("Model-Method Combinations")
                ax.set_ylabel("Mean IoU")
                ax.set_title("Mean IoU: Correct vs Incorrect Classifications")
                ax.legend()

                # Set x-axis labels
                if "model_name" in correct_stats.columns:
                    labels = [
                        f"{row['model_name']}\n{row.get('explainer_name', 'N/A')}"
                        for _, row in correct_stats.iterrows()
                    ]
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=0, fontsize=9)

            # Top-right: Overall IoU distribution
            ax = axes[0, 1]
            if "iou" in self.df.columns:
                sns.boxplot(data=self.df, x="prediction_correct", y="iou", ax=ax)
                ax.set_title("Overall IoU Distribution by Prediction Correctness")
                ax.set_xlabel("Prediction Correct")
                ax.set_ylabel("IoU Score")

            # Bottom-left: IoU by method (if multiple methods exist)
            ax = axes[1, 0]
            if (
                "explainer_name" in self.df.columns
                and self.df["explainer_name"].nunique() > 1
            ):
                sns.violinplot(data=self.df, x="explainer_name", y="iou", ax=ax)
                ax.set_title("IoU Distribution by XAI Method")
                ax.set_xlabel("XAI Method")
                ax.set_ylabel("IoU Score")
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Single method analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title("Method Analysis")

            # Bottom-right: Method performance heatmap
            ax = axes[1, 1]
            if len(correct_stats) > 1:
                # Create performance comparison matrix
                comparison_data = correct_stats.pivot_table(
                    index="model_name",
                    columns="explainer_name",
                    values="mean_iou",
                    fill_value=0,
                )

                if comparison_data.shape[0] > 0 and comparison_data.shape[1] > 0:
                    sns.heatmap(
                        comparison_data,
                        annot=True,
                        fmt=".3f",
                        cmap="RdYlGn",
                        ax=ax,
                        cbar_kws={"label": "Mean IoU (Correct Classifications)"},
                    )
                    ax.set_title(
                        "Mean IoU Performance Matrix\n(Correctly Classified Samples)"
                    )
                    ax.set_xlabel("XAI Method")
                    ax.set_ylabel("Model")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Insufficient data for heatmap",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Single combination analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )

            plt.tight_layout()
            plot_path = save_dir / "iou_distribution_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_paths["iou_distribution_comparison"] = plot_path

        return plot_paths


if __name__ == "__main__":
    with mlflow.start_run(run_name="Comprehensive XAI Meta Analysis"):
        project_root = Path(__file__).resolve().parents[3]
        results_base_dir = project_root / "results"

        # Find all combined datasets for analysis
        combined_datasets = [
            ("resnet_xai_comparison", "combined_data_resnet_xai_methods.csv"),
            ("models_gradcam_comparison2", "combined_data_models_gradcam.csv"),
            ("vgg16_xai_comparison", "combined_data_vgg16_xai_methods.csv"),
        ]

        # Create comprehensive output directory
        comprehensive_output_dir = results_base_dir / "comprehensive_xai_meta_analysis"
        comprehensive_output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        print("=" * 80)
        print("STARTING COMPREHENSIVE XAI META ANALYSIS")
        print("=" * 80)

        for dataset_name, csv_filename in combined_datasets:
            dataset_dir = results_base_dir / dataset_name
            csv_path = dataset_dir / csv_filename

            if not csv_path.exists():
                print(f"WARNING: Skipping {dataset_name}: {csv_path} not found")
                continue

            print(f"\nAnalyzing: {dataset_name}")
            print(f"Data source: {csv_path}")

            # Create output directory for this dataset
            output_dir = comprehensive_output_dir / dataset_name
            meta_analysis_dir = output_dir / "meta_analysis"
            meta_plot_dir = meta_analysis_dir / "plots"
            meta_analysis_dir.mkdir(parents=True, exist_ok=True)
            meta_plot_dir.mkdir(parents=True, exist_ok=True)

            # Load and analyze
            try:
                print(f"Reading CSV from {csv_path.resolve()}")
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
                print(f"Models: {sorted(df['model_name'].unique())}")
                print(f"Explainers: {sorted(df['explainer_name'].unique())}")

                analysis = SingleRunAnalyse(df)

                dataset_results = {
                    "data_path": str(csv_path),
                    "total_samples": len(df),
                    "models": list(df["model_name"].unique()),
                    "explainers": list(df["explainer_name"].unique()),
                }

                print(f"Running analysis for {dataset_name}...")

                # 1. Original Plots erzeugen
                print("Creating metric vs correctness plots...")
                plots = analysis.plot_metric_vs_correctness()
                for name, fig in plots.items():
                    filepath = meta_plot_dir / f"{name}_vs_correctness.png"
                    fig.savefig(filepath, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    mlflow.log_artifact(str(filepath))
                print(f"   Created {len(plots)} metric comparison plots")


                # 3. Scatter-Analyse
                print("Creating scatter analysis...")
                try:
                    grouped_2, fig2 = analysis.scatter("iou")
                    scatter_plot_path = (
                        meta_plot_dir / "prediction_confidence_iou_score.png"
                    )
                    fig2.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig2)
                    mlflow.log_artifact(str(scatter_plot_path))
                    print("   Scatter analysis completed")
                except Exception as e:
                    print(f"   WARNING: Scatter analysis failed: {e}")

                # 4. Neue erweiterte Analysen
                print("Computing F1 scores...")
                f1_scores_df = analysis.calculate_model_method_f1_scores()
                if not f1_scores_df.empty:
                    f1_csv_path = meta_analysis_dir / "f1_scores_by_model_method.csv"
                    f1_scores_df.to_csv(f1_csv_path, index=False)
                    mlflow.log_artifact(str(f1_csv_path))
                    print(f"   F1 scores saved: {len(f1_scores_df)} records")
                    dataset_results["f1_scores"] = f1_scores_df.to_dict("records")
                else:
                    print("   WARNING: No F1 scores computed (insufficient data)")

                print("Computing IoU distribution statistics...")
                iou_stats_df = analysis.calculate_iou_distribution_stats()
                if not iou_stats_df.empty:
                    iou_stats_csv_path = (
                        meta_analysis_dir / "iou_distribution_stats.csv"
                    )
                    iou_stats_df.to_csv(iou_stats_csv_path, index=False)
                    mlflow.log_artifact(str(iou_stats_csv_path))
                    print(
                        f"   IoU distribution stats saved: {len(iou_stats_df)} records"
                    )
                    dataset_results["iou_stats"] = iou_stats_df.to_dict("records")
                else:
                    print("   WARNING: No IoU distribution stats computed")

                print("Creating model/method comparison plots...")
                comparison_plots = analysis.generate_model_method_comparison_plots(
                    meta_plot_dir
                )
                for plot_name, plot_path in comparison_plots.items():
                    mlflow.log_artifact(str(plot_path))
                print(f"   Created {len(comparison_plots)} comparison plots")
                dataset_results["plots_created"] = len(plots) + len(comparison_plots)

                all_results[dataset_name] = dataset_results
                print(f"Analysis completed for {dataset_name}")

            except Exception as e:
                print(f"ERROR: Error analyzing {dataset_name}: {e}")
                continue

        # Create comprehensive summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)

        summary_lines = [
            f"XAI Meta Analysis Summary - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATASETS ANALYZED:",
        ]

        total_samples = 0
        all_models = set()
        all_explainers = set()

        for dataset_name, results in all_results.items():
            summary_lines.extend(
                [
                    "",
                    f"DATASET: {dataset_name.upper()}:",
                    f"   Samples: {results['total_samples']:,}",
                    f"   Models: {', '.join(results['models'])}",
                    f"   Explainers: {', '.join(results['explainers'])}",
                    f"   Plots created: {results.get('plots_created', 0)}",
                ]
            )

            total_samples += results["total_samples"]
            all_models.update(results["models"])
            all_explainers.update(results["explainers"])

            # Add F1 score summary
            if "f1_scores" in results:
                f1_data = results["f1_scores"]
                valid_f1_scores = [
                    item["f1_score"]
                    for item in f1_data
                    if not pd.isna(item["f1_score"])
                ]
                if valid_f1_scores:
                    avg_f1 = np.mean(valid_f1_scores)
                    summary_lines.append(f"   Average F1 Score: {avg_f1:.3f}")

        summary_lines.extend(
            [
                "",
                "OVERALL SUMMARY:",
                f"   Total samples analyzed: {total_samples:,}",
                f"   Unique models: {len(all_models)} ({', '.join(sorted(all_models))})",
                f"   Unique explainers: {len(all_explainers)} ({', '.join(sorted(all_explainers))})",
                f"   Datasets processed: {len(all_results)}",
                "",
                f"Results saved to: {comprehensive_output_dir}",
            ]
        )

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        # Save comprehensive summary
        summary_file = comprehensive_output_dir / "comprehensive_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        mlflow.log_artifact(str(summary_file))

        print("\nCOMPREHENSIVE XAI META ANALYSIS COMPLETED!")
        print(f"All results saved to: {comprehensive_output_dir}")
        print("=" * 80)

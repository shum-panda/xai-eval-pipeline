from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class XaiMetaAnalysis:
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

    def calculate_model_method_f1_scores(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Berechnet F1-Scores für Point Game pro Modell und Methode.
        
        Args:
            threshold: Schwellenwert für Point Game Hit/Miss
            
        Returns:
            DataFrame mit detaillierten F1-Score Metriken
        """
        if 'point_game' not in self.df.columns:
            return pd.DataFrame()
        
        results = []
        
        # Group by model and explainer if available, otherwise just by available columns
        groupby_cols = []
        if 'model_name' in self.df.columns:
            groupby_cols.append('model_name')
        if 'explainer_name' in self.df.columns:
            groupby_cols.append('explainer_name')
        
        if not groupby_cols:
            # Fallback: analyze the entire dataset as one group
            groupby_cols = ['dataset']  # Use a dummy column
            self.df['dataset'] = 'all'
        
        for group_values, group_data in self.df.groupby(groupby_cols):
            if isinstance(group_values, str):
                group_values = [group_values]
            
            # Create group identifier
            group_dict = dict(zip(groupby_cols, group_values))
            
            # Convert point game to binary
            y_true = group_data['prediction_correct'].astype(int)
            y_pred_point_game = (group_data['point_game'] >= threshold).astype(int)
            
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_point_game)) > 1:
                f1 = f1_score(y_true, y_pred_point_game)
                precision = precision_score(y_true, y_pred_point_game)
                recall = recall_score(y_true, y_pred_point_game)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_point_game).ravel()
                
                result = {
                    **group_dict,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'true_positives': tp,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'total_samples': len(group_data),
                    'threshold_used': threshold
                }
            else:
                result = {
                    **group_dict,
                    'f1_score': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'true_positives': np.nan,
                    'true_negatives': np.nan,
                    'false_positives': np.nan,
                    'false_negatives': np.nan,
                    'total_samples': len(group_data),
                    'threshold_used': threshold
                }
            
            results.append(result)
        
        return pd.DataFrame(results)

    def calculate_iou_distribution_stats(self) -> pd.DataFrame:
        """
        Berechnet IoU-Verteilungsstatistiken pro Modell/Methode und Korrektheit.
        
        Returns:
            DataFrame mit IoU-Statistiken aufgeteilt nach Prediction Correctness
        """
        if 'iou' not in self.df.columns:
            return pd.DataFrame()
        
        results = []
        
        # Group by model and explainer if available
        groupby_cols = []
        if 'model_name' in self.df.columns:
            groupby_cols.append('model_name')
        if 'explainer_name' in self.df.columns:
            groupby_cols.append('explainer_name')
        
        if not groupby_cols:
            groupby_cols = ['dataset']
            self.df['dataset'] = 'all'
        
        for group_values, group_data in self.df.groupby(groupby_cols):
            if isinstance(group_values, str):
                group_values = [group_values]
            
            group_dict = dict(zip(groupby_cols, group_values))
            
            # Split by prediction correctness
            correct_data = group_data[group_data['prediction_correct'] == True]['iou']
            incorrect_data = group_data[group_data['prediction_correct'] == False]['iou']
            
            # Calculate statistics for correctly classified
            if len(correct_data) > 0:
                correct_stats = {
                    **group_dict,
                    'prediction_correctness': 'correct',
                    'count': len(correct_data),
                    'mean_iou': correct_data.mean(),
                    'std_iou': correct_data.std(),
                    'median_iou': correct_data.median(),
                    'min_iou': correct_data.min(),
                    'max_iou': correct_data.max(),
                    'q25_iou': correct_data.quantile(0.25),
                    'q75_iou': correct_data.quantile(0.75)
                }
                results.append(correct_stats)
            
            # Calculate statistics for incorrectly classified
            if len(incorrect_data) > 0:
                incorrect_stats = {
                    **group_dict,
                    'prediction_correctness': 'incorrect',
                    'count': len(incorrect_data),
                    'mean_iou': incorrect_data.mean(),
                    'std_iou': incorrect_data.std(),
                    'median_iou': incorrect_data.median(),
                    'min_iou': incorrect_data.min(),
                    'max_iou': incorrect_data.max(),
                    'q25_iou': incorrect_data.quantile(0.25),
                    'q75_iou': incorrect_data.quantile(0.75)
                }
                results.append(incorrect_stats)
        
        return pd.DataFrame(results)

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
        
        # 1. Enhanced F1 Score Comparison with Method Focus
        f1_scores = self.calculate_model_method_f1_scores()
        if not f1_scores.empty:
            # Create separate plots for models with multiple methods
            if 'model_name' in f1_scores.columns and 'explainer_name' in f1_scores.columns:
                models_with_multiple_methods = f1_scores.groupby('model_name')['explainer_name'].nunique()
                models_with_multiple_methods = models_with_multiple_methods[models_with_multiple_methods > 1].index
                
                if len(models_with_multiple_methods) > 0:
                    # Create method comparison plot for models with multiple methods
                    fig, axes = plt.subplots(1, len(models_with_multiple_methods), 
                                           figsize=(8 * len(models_with_multiple_methods), 6))
                    
                    if len(models_with_multiple_methods) == 1:
                        axes = [axes]
                    
                    fig.suptitle('F1 Score Comparison: Methods per Model', fontsize=16, fontweight='bold')
                    
                    for idx, model in enumerate(models_with_multiple_methods):
                        model_data = f1_scores[f1_scores['model_name'] == model]
                        
                        ax = axes[idx]
                        bars = ax.bar(range(len(model_data)), model_data['f1_score'], 
                                     color=plt.cm.viridis(np.linspace(0, 1, len(model_data))))
                        
                        # Add value labels on bars
                        for i, (bar, f1_val) in enumerate(zip(bars, model_data['f1_score'])):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{f1_val:.3f}', ha='center', va='bottom', fontweight='bold')
                        
                        ax.set_title(f'{model}\nMethod Comparison', fontweight='bold')
                        ax.set_ylabel('F1 Score')
                        ax.set_xlabel('XAI Method')
                        ax.set_xticks(range(len(model_data)))
                        ax.set_xticklabels(model_data['explainer_name'], rotation=45)
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(0, max(model_data['f1_score']) * 1.1)
                    
                    plt.tight_layout()
                    plot_path = save_dir / 'f1_score_method_comparison.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths['f1_score_method_comparison'] = plot_path
                
                # General heatmap for all combinations
                fig, ax = plt.subplots(figsize=(12, 8))
                pivot_data = f1_scores.pivot(index='model_name', columns='explainer_name', values='f1_score')
                
                mask = pivot_data.isnull()
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax, 
                           mask=mask, cbar_kws={'label': 'F1 Score'})
                ax.set_title('F1 Scores: Point Game vs Prediction Correctness\n(All Model-Method Combinations)', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('XAI Method')
                ax.set_ylabel('Model')
                
                plt.tight_layout()
                plot_path = save_dir / 'f1_score_heatmap.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['f1_score_heatmap'] = plot_path
        
        # 2. Enhanced IoU Distribution Comparison with Method-specific Analysis
        iou_stats = self.calculate_iou_distribution_stats()
        if not iou_stats.empty:
            # Create comprehensive IoU comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('IoU Analysis: Model-Method Performance Comparison', fontsize=16, fontweight='bold')
            
            # Top-left: Mean IoU comparison
            ax = axes[0, 0]
            correct_stats = iou_stats[iou_stats['prediction_correctness'] == 'correct']
            incorrect_stats = iou_stats[iou_stats['prediction_correctness'] == 'incorrect']
            
            if not correct_stats.empty and not incorrect_stats.empty:
                x = np.arange(len(correct_stats))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, correct_stats['mean_iou'], width, 
                              label='Correctly Classified', alpha=0.8, color='green')
                bars2 = ax.bar(x + width/2, incorrect_stats['mean_iou'], width,
                              label='Incorrectly Classified', alpha=0.8, color='red')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Model-Method Combinations')
                ax.set_ylabel('Mean IoU')
                ax.set_title('Mean IoU: Correct vs Incorrect Classifications')
                ax.legend()
                
                # Set x-axis labels
                if 'model_name' in correct_stats.columns:
                    labels = [f"{row['model_name']}\n{row.get('explainer_name', 'N/A')}" 
                             for _, row in correct_stats.iterrows()]
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=0, fontsize=9)
            
            # Top-right: Overall IoU distribution
            ax = axes[0, 1]
            if 'iou' in self.df.columns:
                sns.boxplot(data=self.df, x='prediction_correct', y='iou', ax=ax)
                ax.set_title('Overall IoU Distribution by Prediction Correctness')
                ax.set_xlabel('Prediction Correct')
                ax.set_ylabel('IoU Score')
            
            # Bottom-left: IoU by method (if multiple methods exist)
            ax = axes[1, 0]
            if 'explainer_name' in self.df.columns and self.df['explainer_name'].nunique() > 1:
                sns.violinplot(data=self.df, x='explainer_name', y='iou', ax=ax)
                ax.set_title('IoU Distribution by XAI Method')
                ax.set_xlabel('XAI Method')
                ax.set_ylabel('IoU Score')
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'Single method analysis', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Method Analysis')
            
            # Bottom-right: Method performance heatmap
            ax = axes[1, 1]
            if len(correct_stats) > 1:
                # Create performance comparison matrix
                comparison_data = correct_stats.pivot_table(
                    index='model_name', columns='explainer_name', values='mean_iou', fill_value=0)
                
                if comparison_data.shape[0] > 0 and comparison_data.shape[1] > 0:
                    sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                               cbar_kws={'label': 'Mean IoU (Correct Classifications)'})
                    ax.set_title('Mean IoU Performance Matrix\n(Correctly Classified Samples)')
                    ax.set_xlabel('XAI Method')
                    ax.set_ylabel('Model')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Single combination analysis', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            plt.tight_layout()
            plot_path = save_dir / 'iou_distribution_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['iou_distribution_comparison'] = plot_path
        
        return plot_paths


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

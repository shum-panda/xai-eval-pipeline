"""
Vereinfachte Analyse-Komponente für XAI-Experimente.
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

from src.analyse.advanced_plotting import AdvancedPlotter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimpleAnalyzer:
    """Vereinfachte Analyse-Klasse für XAI-Experimente."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.logger = logging.getLogger(__name__)
        
    def diagnose_available_data(self) -> None:
        """Zeigt alle verfügbaren Experiment-Daten an."""
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
                        found_experiments.append({
                            'directory': exp_dir.name,
                            'csv_path': csv_file,
                            'samples': len(pd.read_csv(csv_file)),
                            'columns': list(df.columns),
                            'has_model_name': 'model_name' in df.columns,
                            'has_explainer_name': 'explainer_name' in df.columns
                        })
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
                print(f"   [MODEL] Model Name: {'OK' if exp['has_model_name'] else 'MISSING'}")
                print(f"   [EXPLAINER] Explainer Name: {'OK' if exp['has_explainer_name'] else 'MISSING'}")
                print(f"   [COLUMNS] Spalten: {', '.join(exp['columns'][:10])}{'...' if len(exp['columns']) > 10 else ''}")
                print()
        
        print("=" * 80)

    def run_all_analyses(self) -> None:
        """Führt alle Standard-Analysen aus."""
        # ResNet Methodenvergleich
        self.analyze_resnet_methods()

        # VGG Methodenvergleich
        self.analyze_vgg_methods()

        # Modellvergleich mit GradCAM
        self.analyze_model_comparison()

    def analyze_resnet_methods(self) -> None:
        """Analysiert ResNet mit verschiedenen XAI-Methoden."""
        config_names = [
            "config_resnet50_grad_cam",
            "config_resnet50_guided_backprop",
            "config_resnet50_integrated_gradients",
        ]

        output_dir = self.project_root / "results" / "analyse" / "resnet_xai_methods"
        self._run_analysis(config_names, output_dir, "ResNet XAI Methods")

    def analyze_vgg_methods(self) -> None:
        """Analysiert VGG mit verschiedenen XAI-Methoden."""
        config_names = ["config_vgg16_grad_cam", "config_vgg16_score_cam"]

        output_dir = self.project_root / "results" / "analyse"/ "vgg16_xai_methods"
        self._run_analysis(
            config_names, output_dir, "VGG XAI Methods", balance_samples=True
        )

    def analyze_model_comparison(self) -> None:
        """Analysiert verschiedene Modelle mit GradCAM."""
        config_names = [
            "config_resnet18_grad_cam",
            "config_resnet34_grad_cam",
            "config_resnet50_grad_cam",
            "config_vgg16_grad_cam",
        ]

        output_dir = self.project_root / "results" / "analyse"/ "models_gradcam"
        self._run_analysis(config_names, output_dir, "Model Comparison")

    def _run_analysis(
        self,
        config_names: List[str],
        output_dir: Path,
        analysis_name: str,
        balance_samples: bool = False,
    ) -> None:
        """
        Führt eine Analyse für die gegebenen Configs aus.

        Args:
            config_names: Liste der Config-Namen
            output_dir: Ausgabe-Verzeichnis
            analysis_name: Name der Analyse für Logging
            balance_samples: Ob Sample-Sizes balanciert werden sollen
        """
        self.logger.info(f"Starting {analysis_name}")

        # Daten laden
        dfs = []
        for config_name in config_names:
            try:
                df = self._load_experiment_data(config_name)
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} samples from {config_name}")
            except Exception as e:
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
            self.logger.info(f"Created {len(plot_paths)} standard plots for {analysis_name}")
            
            # 2. F1-Score Heatmap
            f1_heatmap_path = self.create_f1_score_heatmap(combined_df, plot_dir, analysis_name)
            if f1_heatmap_path:
                plot_paths['f1_score_heatmap'] = f1_heatmap_path
                
            # 5. Comprehensive histogram comparison
            histogram_comparison_path = self.create_histogram_comparison(combined_df, plot_dir, analysis_name)
            if histogram_comparison_path:
                plot_paths['histogram_comparison'] = histogram_comparison_path
                
            self.logger.info(f"Created {len(plot_paths)} total plots for {analysis_name}")
        except Exception as e:
            self.logger.error(f"Failed to create plots for {analysis_name}: {e}")
            raise

        self.logger.info(f"Completed {analysis_name} - Results in: {output_dir}")

    def _load_experiment_data(self, config_name: str) -> pd.DataFrame:
        """Lädt Experiment-Daten für eine Config aus lokalen results-Verzeichnissen."""
        
        # Definiere mögliche Pfade basierend auf dem Config-Namen
        possible_paths = []
        
        # 1. Direkte Zuordnung basierend auf experiment output_dir
        # Lade die Config um den output_dir zu bestimmen
        try:
            from hydra import compose, initialize_config_dir
            config_dir = self.project_root / "config" / "experiments"
            
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name=f"{config_name}.yaml")
                if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'output_dir'):
                    output_dir = Path(cfg.experiment.output_dir)
                    if not output_dir.is_absolute():
                        output_dir = self.project_root / output_dir
                    possible_paths.append(output_dir / "results_with_metrics.csv")
        except Exception as e:
            self.logger.debug(f"Could not load config {config_name}: {e}")
        
        # 2. Fallback: Suche in results-Verzeichnis basierend auf Config-Namen
        results_dir = self.project_root /"results"
        
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
                        df_test = pd.read_csv(csv_file, nrows=1)
                        possible_paths.append(csv_file)

        
        # Versuche die Pfade in der Reihenfolge
        for csv_path in possible_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        # Füge Metadaten hinzu basierend auf Config-Name
                        model_name, explainer_name = self._extract_model_and_explainer(config_name)
                        if 'model_name' not in df.columns:
                            df['model_name'] = model_name
                        if 'explainer_name' not in df.columns:
                            df['explainer_name'] = explainer_name
                        
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
        """Extrahiert Modell und Explainer aus Config-Namen."""
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
                model_name = name_part[:-len(f"_{explainer}")]
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
        """Balanciert Sample-Größen pro XAI-Methode (nimmt die ersten N pro Gruppe)."""
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

    def create_f1_score_heatmap(self, df: pd.DataFrame, plot_dir: Path, analysis_name: str) -> Path:
        """
        Erstellt eine F1-Score Heatmap für Point Game vs. Prediction Correctness.
        
        Args:
            df: DataFrame mit Experiment-Daten
            plot_dir: Verzeichnis zum Speichern der Plots
            analysis_name: Name der Analyse für den Titel
            
        Returns:
            Pfad zur erstellten Heatmap
        """
        try:
            if "point_game" not in df.columns or "prediction_correct" not in df.columns:
                self.logger.warning("Missing columns for F1-Score heatmap")
                return None
                
            # Ensure prediction_correct is boolean (handle NaN values)
            df = df.copy()  # Work with a copy to avoid SettingWithCopyWarning
            df["prediction_correct"] = df["prediction_correct"].fillna(False).astype(bool)
            
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
                        group_cols[0]: group_values[0] if len(group_values) > 0 else "Unknown"
                    }
                    if len(group_cols) > 1:
                        result[group_cols[1]] = group_values[1] if len(group_values) > 1 else "Unknown"
                        
                    result.update({
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "samples": len(group_data)
                    })
                    f1_results.append(result)
            
            if not f1_results:
                self.logger.warning("No valid F1 scores computed")
                return None
                
            f1_df = pd.DataFrame(f1_results)
            
            # Create heatmap
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # F1-Score Heatmap
            if len(group_cols) == 2:
                pivot_f1 = f1_df.pivot(index=group_cols[0], columns=group_cols[1], values="f1_score")
                pivot_precision = f1_df.pivot(index=group_cols[0], columns=group_cols[1], values="precision")
                pivot_recall = f1_df.pivot(index=group_cols[0], columns=group_cols[1], values="recall")
            else:
                # Single grouping column - create simple bar plot instead
                fig, ax = plt.subplots(figsize=(10, 6))
                f1_df.plot(x=group_cols[0], y="f1_score", kind="bar", ax=ax, color="viridis")
                ax.set_title(f"F1 Scores: {analysis_name}")
                ax.set_ylabel("F1 Score")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plot_path = plot_dir / f"f1_score_heatmap_{analysis_name.lower().replace(' ', '_')}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
                plt.close()
                return plot_path
            
            # F1 Score
            sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[0], 
                       cbar_kws={"label": "F1 Score"}, vmin=0, vmax=1)
            axes[0].set_title("F1 Scores", fontweight="bold")
            axes[0].set_ylabel("Model")
            axes[0].set_xlabel("XAI Method")
            
            # Precision
            sns.heatmap(pivot_precision, annot=True, fmt=".3f", cmap="Blues", ax=axes[1], 
                       cbar_kws={"label": "Precision"}, vmin=0, vmax=1)
            axes[1].set_title("Precision", fontweight="bold")
            axes[1].set_ylabel("Model")
            axes[1].set_xlabel("XAI Method")
            
            # Recall
            sns.heatmap(pivot_recall, annot=True, fmt=".3f", cmap="Oranges", ax=axes[2], 
                       cbar_kws={"label": "Recall"}, vmin=0, vmax=1)
            axes[2].set_title("Recall", fontweight="bold")
            axes[2].set_ylabel("Model")
            axes[2].set_xlabel("XAI Method")
            
            plt.suptitle(f"Performance Metrics: {analysis_name}\n(Point Game vs. Prediction Correctness)", 
                        fontsize=16, fontweight="bold")
            plt.tight_layout()
            
            plot_path = plot_dir / f"f1_score_heatmap_{analysis_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()
            
            # Save F1 data as CSV
            csv_path = plot_dir.parent / f"f1_scores_{analysis_name.lower().replace(' ', '_')}.csv"
            f1_df.to_csv(csv_path, index=False)
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to create F1-Score heatmap: {e}")
            raise

    def create_histogram_comparison(self, df: pd.DataFrame, plot_dir: Path, analysis_name: str) -> Path:
        """
        Erstellt einen umfassenden Histogram-Vergleich aller Metriken.
        
        Args:
            df: DataFrame mit Experiment-Daten
            plot_dir: Verzeichnis zum Speichern der Plots  
            analysis_name: Name der Analyse für den Titel
            
        Returns:
            Pfad zum erstellten Histogram-Vergleich
        """
        try:
            # Define available metrics (check for different column naming patterns)
            metrics_map = {
                "iou": ["iou"],
                "pixel_precision": ["pixel_precision", "pixelprecisionrecall_precision"], 
                "pixel_recall": ["pixel_recall", "pixelprecisionrecall_recall"],
                "point_game": ["point_game"],
                "prediction_confidence": ["prediction_confidence"]
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
            
            plt.style.use('seaborn-v0_8')
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
                    correct_data = df[df["prediction_correct"] == True][actual_col].dropna()
                    incorrect_data = df[df["prediction_correct"] == False][actual_col].dropna()
                    
                    # Create overlapping histograms
                    bins = np.linspace(0, 1, 31) if metric in ["iou", "pixel_precision", "pixel_recall", "point_game"] else 25
                    
                    if len(correct_data) > 0:
                        ax.hist(correct_data, bins=bins, alpha=0.7, color=colors[0], 
                               label=f"{labels[0]} (n={len(correct_data)})", density=True,
                               edgecolor='white', linewidth=0.5)
                        
                        # Add mean line
                        mean_correct = correct_data.mean()
                        ax.axvline(mean_correct, color=colors[0], linestyle='--', linewidth=2, alpha=0.8)
                        ax.text(mean_correct, ax.get_ylim()[1] * 0.9, f'μ={mean_correct:.3f}', 
                               rotation=90, color=colors[0], fontweight='bold')
                    
                    if len(incorrect_data) > 0:
                        ax.hist(incorrect_data, bins=bins, alpha=0.7, color=colors[1], 
                               label=f"{labels[1]} (n={len(incorrect_data)})", density=True,
                               edgecolor='white', linewidth=0.5)
                        
                        # Add mean line
                        mean_incorrect = incorrect_data.mean()
                        ax.axvline(mean_incorrect, color=colors[1], linestyle='--', linewidth=2, alpha=0.8)
                        ax.text(mean_incorrect, ax.get_ylim()[1] * 0.8, f'μ={mean_incorrect:.3f}', 
                               rotation=90, color=colors[1], fontweight='bold')
                        
                    # Statistical tests if both groups have data
                    if len(correct_data) > 0 and len(incorrect_data) > 0:
                        from scipy import stats
                        
                        # Mann-Whitney U test (non-parametric)
                        mw_statistic, mw_p_value = stats.mannwhitneyu(correct_data, incorrect_data, alternative='two-sided')
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(correct_data) - 1) * correct_data.std()**2 + 
                                            (len(incorrect_data) - 1) * incorrect_data.std()**2) / 
                                           (len(correct_data) + len(incorrect_data) - 2))
                        cohens_d = (correct_data.mean() - incorrect_data.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        # Kolmogorov-Smirnov test for distribution differences
                        ks_statistic, ks_p_value = stats.ks_2samp(correct_data, incorrect_data)
                        
                        # Create statistical summary
                        stat_text = f'Mann-Whitney U: p={mw_p_value:.3e}\nCohen\'s d: {cohens_d:.3f}\nKS test: p={ks_p_value:.3e}'
                        
                        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, 
                               fontsize=9, verticalalignment='top', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
                else:
                    # Single histogram if no correctness data - use actual column name
                    actual_col = metric_name_map.get(metric, metric)
                    ax.hist(df[actual_col].dropna(), bins=25, alpha=0.7, color="skyblue", 
                           density=True, edgecolor='white', linewidth=0.5)
                    
                    mean_val = df[actual_col].mean()
                    ax.axvline(mean_val, color="blue", linestyle='--', linewidth=2, alpha=0.8)
                    ax.text(mean_val, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.3f}', 
                           rotation=90, color="blue", fontweight='bold')
                
                # Styling
                ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
                ax.set_xlabel(f"{metric.replace('_', ' ').title()} Score", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f8f8')
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f"Metric Distribution Analysis: {analysis_name}\nPerformance by Prediction Correctness", 
                        fontsize=16, fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            
            plot_path = plot_dir / f"histogram_comparison_{analysis_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
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
    
    print("\n" + "="*50)
    print("Starte jetzt die Analysen...")
    print("="*50)
    
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()

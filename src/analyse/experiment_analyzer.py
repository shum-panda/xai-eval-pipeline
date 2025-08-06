import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hydra import compose, initialize
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.analyse.advanced_plotting import AdvancedPlotter
from src.analyse.experiment_collection import ExperimentCollection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "config" / "experiments"
RESULT_FILENAME = "results_with_metrics.csv"


class AnalysisMode(Enum):
    """Verschiedene Analyse-Modi für unterschiedliche Experimentvergleiche."""

    RESNET_XAI_METHODS = "resnet_xai_methods"  # ResNet mit verschiedenen XAI-Methoden
    MODELS_GRADCAM = "models_gradcam"  # Verschiedene Modelle mit GradCAM
    VGG16_XAI_METHODS = "vgg16_xai_methods"  # VGG16 mit verschiedenen XAI-Methoden


class ExperimentAnalyzer:
    """
    Flexibler Experiment-Analyzer für verschiedene Analyse-Szenarien.

    Unterstützt drei Hauptmodi:
    1. ResNet + verschiedene XAI-Methoden
    2. Verschiedene Modelle + GradCAM
    3. VGG16 + verschiedene XAI-Methoden (GradCAM vs ScoreCAM)
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_dir = EXPERIMENT_CONFIG_DIR
        self.logger = logging.getLogger(__name__)

        # Konfigurierbare Experiment-Listen
        self.experiment_configs = {
            AnalysisMode.RESNET_XAI_METHODS: [
                "config_resnet50_grad_cam",
                "config_resnet50_guided_backprop",
                "config_resnet50_integrated_gradients",
            ],
            AnalysisMode.MODELS_GRADCAM: [
                "config_resnet18_grad_cam",
                "config_resnet50_grad_cam",
                "config_resnet101_grad_cam",
                "config_vgg16_grad_cam",
            ],
            AnalysisMode.VGG16_XAI_METHODS: [
                "config_vgg16_grad_cam",
                "config_vgg16_score_cam",
            ],
        }

    def update_experiment_config(
        self, mode: AnalysisMode, config_names: List[str]
    ) -> None:
        """
        Aktualisiere die Experiment-Konfiguration für einen bestimmten Modus.

        Args:
            mode: Analyse-Modus
            config_names: Liste der Config-Namen (ohne .yaml Extension)
        """
        self.experiment_configs[mode] = config_names
        print(f"[OK] Updated {mode.value} experiments: {config_names}")

    def get_available_configs(self) -> List[str]:
        """
        Findet alle verfügbaren Experiment-Configs im Config-Verzeichnis.

        Returns:
            Liste aller verfügbaren Config-Namen
        """
        config_files = []
        # Flexibleres Pattern das mehrteilige Namen erlaubt
        pattern = re.compile(r"^config_.+\.ya?ml$")

        for file in self.config_dir.iterdir():
            if pattern.match(file.name):
                config_name = file.stem  # ohne ".yaml"
                # Zusätzliche Validierung durch extract_model_and_explainer
                try:
                    self.extract_model_and_explainer(config_name)
                    config_files.append(config_name)
                except ValueError:
                    # Überspringe invalide Config-Namen
                    continue
        return sorted(config_files)

    def extract_model_and_explainer(self, config_name: str) -> tuple[str, str]:
        """
        Extrahiert Modell und Explainer aus Config-Namen.

        Args:
            config_name: Config-Name wie "config_resnet50_grad_cam" oder "config_vgg16_score_cam"

        Returns:
            Tuple (model_name, explainer_name)
        """
        # Entferne "config_" Präfix
        if config_name.startswith("config_"):
            name_part = config_name[7:]  # Entferne "config_"
        else:
            name_part = config_name

        # Spezielle Behandlung für bekannte Explainer-Namen
        known_explainers = [
            "grad_cam",
            "score_cam",
            "guided_backprop",
            "integrated_gradients",
        ]

        # Finde den längsten passenden Explainer am Ende
        explainer_name = None
        model_name = None

        for explainer in known_explainers:
            if name_part.endswith(f"_{explainer}"):
                explainer_name = explainer
                model_name = name_part[: -len(f"_{explainer}")]
                break

        if not explainer_name or not model_name:
            # Fallback: Standard Pattern für einfache Fälle
            pattern = re.compile(r"^(\w+)_(\w+)$")
            match = pattern.match(name_part)
            if match:
                model_name = match.group(1)
                explainer_name = match.group(2)
            else:
                raise ValueError(f"Invalid config name: {config_name}")

        return model_name, explainer_name

    def load_experiment_data(self, config_name: str) -> pd.DataFrame:
        """
        Lädt Experiment-Daten für eine bestimmte Config.

        Args:
            config_name: Name der Config-Datei

        Returns:
            DataFrame mit Experiment-Daten
        """
        with initialize(config_path="../../config/experiments", version_base=None):
            cfg = compose(config_name=config_name)

        output_dir = self.project_root / Path(cfg.experiment.output_dir)
        csv_path = output_dir / RESULT_FILENAME

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing result CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        model_name, explainer_name = self.extract_model_and_explainer(config_name)

        # Metadaten hinzufügen
        df["model_name"] = model_name
        df["explainer_name"] = explainer_name
        df["experiment_name"] = cfg.experiment.name
        df["config_name"] = config_name

        return df

    def load_experiments_for_mode(self, mode: AnalysisMode) -> ExperimentCollection:
        """
        Lädt alle Experimente für einen bestimmten Analyse-Modus.

        Args:
            mode: Analyse-Modus

        Returns:
            ExperimentCollection mit geladenen Daten
        """
        config_names = self.experiment_configs.get(mode, [])

        if not config_names:
            raise ValueError(f"No experiments configured for mode: {mode.value}")

        dfs = []
        failed_configs = []

        for config_name in config_names:
            try:
                self.logger.debug(f"Loading experiment data: {config_name}")
                df = self.load_experiment_data(config_name)
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} samples from {config_name}")
            except Exception as e:
                self.logger.error(f"Failed to load {config_name}: {e}")
                failed_configs.append(config_name)

        if not dfs:
            raise RuntimeError(f"No experiments could be loaded for mode: {mode.value}")

        if failed_configs:
            self.logger.warning(f"Failed to load configs: {failed_configs}")

        collection = ExperimentCollection(dfs)
        self.logger.info(f"Created collection with {len(collection.df)} total samples")

        # Balance sample sizes for fair comparison if this is VGG method comparison
        if mode == AnalysisMode.VGG16_XAI_METHODS:
            collection = self.balance_sample_sizes(collection)
            self.logger.info(
                f"Balanced VGG collection to {len(collection.df)} total samples"
            )

        return collection

    def analyze_mode(
        self, mode: AnalysisMode, output_subdir: Optional[str] = None
    ) -> Dict:
        """
        Führt eine vollständige Analyse für einen bestimmten Modus durch.

        Args:
            mode: Analyse-Modus
            output_subdir: Optionales Unterverzeichnis für Outputs

        Returns:
            Dictionary mit Analyse-Ergebnissen
        """
        self.logger.info(f"Starting analysis for mode: {mode.value}")
        self.logger.debug(f"Experiments: {self.experiment_configs[mode]}")

        # Daten laden
        collection = self.load_experiments_for_mode(mode)

        # Output-Verzeichnis vorbereiten
        if output_subdir:
            output_dir = self.project_root / "results" / output_subdir
        else:
            output_dir = self.project_root / "results" / f"analysis_{mode.value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Basis-Analysen durchführen
        results = {}

        # 1. Accuracy-Analyse
        self.logger.info("Computing accuracy metrics")
        df_accuracy = collection.group_accuracy_by_model()
        results["accuracy"] = df_accuracy

        accuracy_path = output_dir / "accuracy_by_model_explainer.csv"
        df_accuracy.to_csv(accuracy_path, index=False)
        self.logger.debug(f"Accuracy saved: {accuracy_path}")

        # 2. Korrelationsmatrix
        self.logger.info("Computing correlation matrix")
        correlation_matrix = collection.correlation_matrix_all_metrics()
        results["correlation"] = correlation_matrix

        corr_path = output_dir / "correlation_matrix.csv"
        correlation_matrix.to_csv(corr_path)
        self.logger.debug(f"Correlation matrix saved: {corr_path}")

        # 3. Vollständige Daten exportieren
        combined_path = output_dir / f"combined_data_{mode.value}.csv"
        collection.df.to_csv(combined_path, index=False)
        self.logger.debug(f"Combined data saved: {combined_path}")

        # 4. Point Game F1 Score Analysis
        if "point_game" in collection.df.columns:
            self.logger.info("Computing Point Game F1 scores")
            f1_scores_df = self.calculate_point_game_f1_scores(collection)
            results["point_game_f1_scores"] = f1_scores_df

            f1_path = output_dir / "point_game_f1_scores.csv"
            f1_scores_df.to_csv(f1_path, index=False)
            self.logger.debug(f"Point Game F1 scores saved: {f1_path}")

        # 5. Erweiterte Plots mit AdvancedPlotter
        self.logger.info("Creating comprehensive visualizations")

        try:
            advanced_plotter = AdvancedPlotter(collection.df, plot_dir)
            advanced_plot_paths = advanced_plotter.create_comprehensive_analysis()
            results["advanced_plots"] = advanced_plot_paths
            self.logger.info(f"Created {len(advanced_plot_paths)} advanced plots")
        except Exception as e:
            self.logger.warning(f"Failed to create advanced plots: {e}")
            results["advanced_plots"] = {}

        # 5. Klassische Metrik-Plots (als Fallback)
        metrics_to_plot = [
            "iou",
            "prediction_confidence",
            "point_game",
            "pixelprecisionrecall_precision",
            "pixelprecisionrecall_recall",
        ]

        self.logger.info(f"Creating {len(metrics_to_plot)} classic metric plots")
        classic_plot_paths = {}

        for metric in metrics_to_plot:
            try:
                plot_path = collection.plot_metric_comparison(
                    metric=metric, save_dir=plot_dir
                )
                classic_plot_paths[metric] = plot_path
                self.logger.debug(f"Classic plot created: {metric}")
            except Exception as e:
                self.logger.warning(f"Failed to plot {metric}: {e}")

        results["classic_plots"] = classic_plot_paths

        # 5. Zusammenfassung erstellen
        summary = self._create_summary(collection, mode)
        results["summary"] = summary

        summary_path = output_dir / "analysis_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        self.logger.debug(f"Summary saved: {summary_path}")

        self.logger.info(f"Analysis complete for {mode.value}")
        self.logger.info(f"Results saved in: {output_dir}")

        return results

    def balance_sample_sizes(
        self, collection: ExperimentCollection
    ) -> ExperimentCollection:
        """
        Balanciert die Anzahl der Samples pro XAI-Methode für faire Vergleiche.
        Reduziert alle Gruppen auf die Größe der kleinsten Gruppe.

        Args:
            collection: ExperimentCollection mit unbalancierten Daten

        Returns:
            ExperimentCollection mit balancierten Sample-Größen
        """
        df = collection.df.copy()

        # Anzahl Samples pro Explainer-Methode ermitteln
        group_counts = df.groupby("explainer_name").size()
        self.logger.info(f"Original sample counts per method: {group_counts.to_dict()}")

        # Kleinste Gruppe finden
        min_samples = group_counts.min()
        self.logger.info(f"Balancing to {min_samples} samples per method")

        # Für jede Explainer-Methode nur n=min_samples Samples behalten
        balanced_dfs = []

        for explainer_name in df["explainer_name"].unique():
            explainer_data = df[df["explainer_name"] == explainer_name]

            # Zufällige Auswahl von min_samples
            if len(explainer_data) > min_samples:
                sampled_data = explainer_data.sample(n=min_samples, random_state=42)
                self.logger.info(
                    f"Reduced {explainer_name} from {len(explainer_data)} to {len(sampled_data)} samples"
                )
            else:
                sampled_data = explainer_data
                self.logger.info(
                    f"Kept all {len(sampled_data)} samples for {explainer_name}"
                )

            balanced_dfs.append(sampled_data)

        # Zusammenführen
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

        # Neue ExperimentCollection erstellen
        balanced_collection = ExperimentCollection([balanced_df])

        # Finale Statistiken
        final_counts = balanced_collection.df.groupby("explainer_name").size()
        self.logger.info(f"Final balanced counts: {final_counts.to_dict()}")

        return balanced_collection

    def calculate_point_game_f1_scores(
        self, collection: ExperimentCollection, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Berechnet F1-Scores für Point Game Metric nach Modell und Explainer.

        Args:
            collection: ExperimentCollection mit den Daten
            threshold: Schwellenwert für Point Game Hit/Miss (default: 0.5)

        Returns:
            DataFrame mit F1-Scores pro Modell/Explainer Kombination
        """
        df = collection.df

        if "point_game" not in df.columns or "prediction_correct" not in df.columns:
            self.logger.warning(
                "Point game or prediction correctness data not available"
            )
            return pd.DataFrame()

        results = []

        for model in df["model_name"].unique():
            for explainer in df["explainer_name"].unique():
                subset = df[
                    (df["model_name"] == model) & (df["explainer_name"] == explainer)
                ]

                if len(subset) == 0:
                    continue

                # Convert point game to binary (hit=1, miss=0)
                y_true = subset["prediction_correct"].astype(int)
                y_pred_point_game = (subset["point_game"] >= threshold).astype(int)

                if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_point_game)) > 1:
                    # Calculate metrics only if both classes are present
                    f1 = f1_score(y_true, y_pred_point_game)
                    precision = precision_score(y_true, y_pred_point_game)
                    recall = recall_score(y_true, y_pred_point_game)

                    # Get confusion matrix components
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_point_game).ravel()

                    results.append(
                        {
                            "model_name": model,
                            "explainer_name": explainer,
                            "f1_score": f1,
                            "precision": precision,
                            "recall": recall,
                            "true_positives": tp,
                            "true_negatives": tn,
                            "false_positives": fp,
                            "false_negatives": fn,
                            "total_samples": len(subset),
                            "threshold_used": threshold,
                        }
                    )
                else:
                    # Handle edge cases where only one class is present
                    results.append(
                        {
                            "model_name": model,
                            "explainer_name": explainer,
                            "f1_score": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "true_positives": np.nan,
                            "true_negatives": np.nan,
                            "false_positives": np.nan,
                            "false_negatives": np.nan,
                            "total_samples": len(subset),
                            "threshold_used": threshold,
                        }
                    )

        return pd.DataFrame(results)

    def _create_summary(
        self, collection: ExperimentCollection, mode: AnalysisMode
    ) -> str:
        """Erstellt eine Textual-Zusammenfassung der Analyse."""
        df = collection.df

        summary_lines = [
            f"=== ANALYSIS SUMMARY: {mode.value.upper()} ===",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[CHART] DATASET OVERVIEW:",
            f"Total samples: {len(df):,}",
            f"Unique models: {df['model_name'].nunique()}",
            f"Unique explainers: {df['explainer_name'].nunique()}",
            f"Models: {sorted(df['model_name'].unique())}",
            f"Explainers: {sorted(df['explainer_name'].unique())}",
            "",
            "[TARGET] ACCURACY OVERVIEW:",
        ]

        # Accuracy pro Kombination
        acc_df = collection.group_accuracy_by_model()
        for _, row in acc_df.iterrows():
            summary_lines.append(
                f"  {row['model_name']} + {row['explainer_name']}: "
                f"{row['accuracy']:.3f}"
            )

        # Point Game F1 Scores
        if "point_game" in df.columns:
            summary_lines.extend(
                [
                    "",
                    "[MEDAL] POINT GAME F1 SCORES:",
                ]
            )

            f1_scores = self.calculate_point_game_f1_scores(collection)
            for _, row in f1_scores.iterrows():
                if not np.isnan(row["f1_score"]):
                    summary_lines.append(
                        f"  {row['model_name']} + {row['explainer_name']}: "
                        f"F1={row['f1_score']:.3f}, "
                        f"Precision={row['precision']:.3f}, "
                        f"Recall={row['recall']:.3f}"
                    )
                else:
                    summary_lines.append(
                        f"  {row['model_name']} + {row['explainer_name']}: "
                        "F1=N/A (insufficient data)"
                    )

        summary_lines.extend(
            [
                "",
                "[TRENDING_UP] METRIC STATISTICS:",
            ]
        )

        # Metrik-Statistiken
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        metric_cols = [
            col
            for col in numeric_cols
            if col
            in [
                "iou",
                "point_game",
                "prediction_confidence",
                "pixelprecisionrecall_precision",
                "pixelprecisionrecall_recall",
            ]
        ]

        for metric in metric_cols:
            if metric in df.columns:
                summary_lines.append(
                    f"  {metric}: mean={df[metric].mean():.3f}, "
                    f"std={df[metric].std():.3f}, "
                    f"min={df[metric].min():.3f}, "
                    f"max={df[metric].max():.3f}"
                )

        return "\n".join(summary_lines)

    def run_all_analyses(self) -> Dict:
        """
        Führt alle verfügbaren Analysen durch.

        Returns:
            Dictionary mit allen Analyse-Ergebnissen
        """
        all_results = {}

        for mode in AnalysisMode:
            try:
                results = self.analyze_mode(mode)
                all_results[mode.value] = results
            except Exception as e:
                self.logger.error(f"Failed analysis for {mode.value}: {e}")
                all_results[mode.value] = {"error": str(e)}

        return all_results

    def print_available_configs(self) -> None:
        """Druckt alle verfügbaren Konfigurationen."""
        available = self.get_available_configs()

        print("\n[SEARCH] AVAILABLE EXPERIMENT CONFIGS:")
        for config in available:
            try:
                model, explainer = self.extract_model_and_explainer(config)
                print(f"  {config} -> {model} + {explainer}")
            except ValueError:
                print(f"  {config} -> [invalid format]")

        print(f"\nTotal: {len(available)} configs found")

    def print_current_setup(self) -> None:
        """Druckt die aktuelle Experiment-Konfiguration."""
        print("\n[GEAR] CURRENT EXPERIMENT SETUP:")

        for mode, configs in self.experiment_configs.items():
            print(f"\n{mode.value.upper()}:")
            for config in configs:
                try:
                    model, explainer = self.extract_model_and_explainer(config)
                    print(f"  [OK] {config} -> {model} + {explainer}")
                except ValueError:
                    print(f"  [WARNING] {config} -> [invalid format]")


if __name__ == "__main__":
    analyzer = ExperimentAnalyzer()

    # Zeige verfügbare Configs
    analyzer.print_available_configs()

    # Zeige aktuelle Setup
    analyzer.print_current_setup()

    # Führe alle Analysen durch
    print("\n[ROCKET] STARTING ALL ANALYSES...")
    results = analyzer.run_all_analyses()

    print("\n[PARTY] ALL ANALYSES COMPLETE!")
    for mode, result in results.items():
        if "error" in result:
            print(f"[ERROR] {mode}: {result['error']}")
        else:
            print(f"[OK] {mode}: Successfully completed")

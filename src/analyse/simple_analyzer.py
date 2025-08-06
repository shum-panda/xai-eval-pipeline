"""
Vereinfachte Analyse-Komponente für XAI-Experimente.
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.analyse.advanced_plotting import AdvancedPlotter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimpleAnalyzer:
    """Vereinfachte Analyse-Klasse für XAI-Experimente."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.logger = logging.getLogger(__name__)

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

        output_dir = self.project_root / "results" / "resnet_xai_methods"
        self._run_analysis(config_names, output_dir, "ResNet XAI Methods")

    def analyze_vgg_methods(self) -> None:
        """Analysiert VGG mit verschiedenen XAI-Methoden."""
        config_names = ["config_vgg16_grad_cam", "config_vgg16_score_cam"]

        output_dir = self.project_root / "results" / "vgg16_xai_methods"
        self._run_analysis(
            config_names, output_dir, "VGG XAI Methods", balance_samples=True
        )

    def analyze_model_comparison(self) -> None:
        """Analysiert verschiedene Modelle mit GradCAM."""
        config_names = [
            "config_resnet18_grad_cam",
            "config_resnet50_grad_cam",
            "config_vgg16_grad_cam",
        ]

        output_dir = self.project_root / "results" / "models_gradcam"
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
                self.logger.error(f"Failed to load {config_name}: {e}")

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

        # Plots erstellen
        try:
            plotter = AdvancedPlotter(combined_df, plot_dir)
            plot_paths = plotter.create_comprehensive_analysis()
            self.logger.info(f"Created {len(plot_paths)} plots for {analysis_name}")
        except Exception as e:
            self.logger.error(f"Failed to create plots for {analysis_name}: {e}")

        self.logger.info(f"Completed {analysis_name} - Results in: {output_dir}")

    def _load_experiment_data(self, config_name: str) -> pd.DataFrame:
        """Lädt Experiment-Daten für eine Config."""
        # Suche nach MLflow Experiment-Ordnern
        mlruns_dir = self.project_root / "mlruns"

        for experiment_dir in mlruns_dir.glob("*/"):
            if experiment_dir.name in ["0", ".trash"]:
                continue

            for run_dir in experiment_dir.glob("*/"):
                artifacts_dir = run_dir / "artifacts"
                if artifacts_dir.exists():
                    # Prüfe ob es der richtige Config-Name ist
                    tags_file = run_dir / "tags"
                    if tags_file.exists():
                        try:
                            with open(tags_file) as f:
                                content = f.read()
                                if config_name in content:
                                    # Lade die Ergebnisse
                                    results_file = (
                                        artifacts_dir / "results_with_metrics.csv"
                                    )
                                    if results_file.exists():
                                        return pd.read_csv(results_file)
                        except:
                            continue

        raise FileNotFoundError(f"No results found for {config_name}")

    def _balance_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balanciert Sample-Größen pro XAI-Methode."""
        group_counts = df.groupby("explainer_name").size()
        min_samples = group_counts.min()

        self.logger.info(f"Balancing to {min_samples} samples per method")

        balanced_dfs = []
        for explainer_name in df["explainer_name"].unique():
            explainer_data = df[df["explainer_name"] == explainer_name]
            if len(explainer_data) > min_samples:
                sampled_data = explainer_data.sample(n=min_samples, random_state=42)
            else:
                sampled_data = explainer_data
            balanced_dfs.append(sampled_data)

        return pd.concat(balanced_dfs, ignore_index=True)


def main():
    """Führt alle Analysen aus."""
    analyzer = SimpleAnalyzer()
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()

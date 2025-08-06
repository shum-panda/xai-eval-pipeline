import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.analyse.experiment_analyzer import AnalysisMode, ExperimentAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigBasedAnalyzer(ExperimentAnalyzer):
    """
    Erweiterte Version des ExperimentAnalyzers, die YAML-Konfigurationsdateien verwendet.
    """

    def __init__(self, config_file: Optional[Path] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if config_file is None:
            config_file = Path(__file__).parent / "experiment_configs.yaml"

        self.config_file = config_file
        self.load_config_from_yaml()

    def load_config_from_yaml(self) -> None:
        """Lädt Experiment-Konfigurationen aus YAML-Datei."""
        if not self.config_file.exists():
            self.logger.warning(f"Config file not found: {self.config_file}")
            self.logger.info("Using default configuration")
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Update experiment configs basierend auf YAML
            if "resnet_xai_methods" in config_data:
                experiments = config_data["resnet_xai_methods"]["experiments"]
                self.experiment_configs[AnalysisMode.RESNET_XAI_METHODS] = experiments

            if "models_gradcam" in config_data:
                experiments = config_data["models_gradcam"]["experiments"]
                self.experiment_configs[AnalysisMode.MODELS_GRADCAM] = experiments

            if "vgg16_xai_methods" in config_data:
                experiments = config_data["vgg16_xai_methods"]["experiments"]
                self.experiment_configs[AnalysisMode.VGG16_XAI_METHODS] = experiments

            self.logger.info(f"Configuration loaded from: {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.logger.info("Using default configuration")

    def get_output_subdir(self, mode_key: str) -> Optional[str]:
        """Ermittelt das Output-Untervezeichnis aus der YAML-Config."""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if mode_key in config_data and "output_subdir" in config_data[mode_key]:
                return config_data[mode_key]["output_subdir"]

        except Exception:
            pass

        return None

    def analyze_resnet_xai_methods(self) -> Dict:
        """Analysiert ResNet mit verschiedenen XAI-Methoden."""
        mode = AnalysisMode.RESNET_XAI_METHODS
        output_subdir = self.get_output_subdir("resnet_xai_methods")

        self.logger.info("Starting ResNet + XAI methods analysis")

        return self.analyze_mode(mode, output_subdir)

    def analyze_models_gradcam(self) -> Dict:
        """Analysiert verschiedene Modelle mit GradCAM."""
        mode = AnalysisMode.MODELS_GRADCAM
        output_subdir = self.get_output_subdir("models_gradcam")

        self.logger.info("Starting Models + GradCAM analysis")

        return self.analyze_mode(mode, output_subdir)

    def analyze_vgg16_xai_methods(self) -> Dict:
        """Analysiert VGG16 mit verschiedenen XAI-Methoden (GradCAM vs ScoreCAM)."""
        mode = AnalysisMode.VGG16_XAI_METHODS
        output_subdir = self.get_output_subdir("vgg16_xai_methods")

        self.logger.info("Starting VGG16 + XAI methods analysis")

        return self.analyze_mode(mode, output_subdir)

    def run_specific_analysis(self, analysis_name: str) -> Dict:
        """
        Führt eine spezifische Analyse basierend auf dem Namen durch.

        Args:
            analysis_name: Name der Analyse ('resnet_xai', 'models_gradcam', oder 'vgg16_xai')
        """
        if analysis_name == "resnet_xai":
            return self.analyze_resnet_xai_methods()
        elif analysis_name == "models_gradcam":
            return self.analyze_models_gradcam()
        elif analysis_name == "vgg16_xai":
            return self.analyze_vgg16_xai_methods()
        else:
            raise ValueError(f"Unknown analysis: {analysis_name}")

    def update_config_file(self, new_config: Dict) -> None:
        """
        Aktualisiert die YAML-Konfigurationsdatei.

        Args:
            new_config: Neue Konfiguration als Dictionary
        """
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    new_config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
            self.logger.info(f"Configuration updated: {self.config_file}")

            # Reload config
            self.load_config_from_yaml()

        except Exception as e:
            self.logger.error(f"Error updating config: {e}")

    def add_experiment_to_mode(self, mode_key: str, experiment_name: str) -> None:
        """
        Fügt ein Experiment zu einem bestimmten Modus hinzu.

        Args:
            mode_key: Schlüssel des Modus ('resnet_xai_methods', 'models_gradcam', oder 'vgg16_xai_methods')
            experiment_name: Name des Experiments (z.B. 'config_resnet18_grad_cam')
        """
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if mode_key not in config_data:
                self.logger.error(f"Mode '{mode_key}' not found in config")
                return

            if experiment_name not in config_data[mode_key]["experiments"]:
                config_data[mode_key]["experiments"].append(experiment_name)
                self.update_config_file(config_data)
                self.logger.info(f"Added {experiment_name} to {mode_key}")
            else:
                self.logger.warning(f"{experiment_name} already in {mode_key}")

        except Exception as e:
            self.logger.error(f"Error adding experiment: {e}")

    def remove_experiment_from_mode(self, mode_key: str, experiment_name: str) -> None:
        """
        Entfernt ein Experiment aus einem bestimmten Modus.

        Args:
            mode_key: Schlüssel des Modus
            experiment_name: Name des zu entfernenden Experiments
        """
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if mode_key not in config_data:
                self.logger.error(f"Mode '{mode_key}' not found in config")
                return

            if experiment_name in config_data[mode_key]["experiments"]:
                config_data[mode_key]["experiments"].remove(experiment_name)
                self.update_config_file(config_data)
                self.logger.info(f"Removed {experiment_name} from {mode_key}")
            else:
                self.logger.warning(f"{experiment_name} not found in {mode_key}")

        except Exception as e:
            self.logger.error(f"Error removing experiment: {e}")


def main():
    """Führt die umfassende Analyse aus."""
    analyzer = ConfigBasedAnalyzer()

    # Alle Analysen ausführen
    analyzer.analyze_resnet_xai_methods()
    analyzer.analyze_models_gradcam()
    analyzer.analyze_vgg16_xai_methods()


if __name__ == "__main__":
    main()

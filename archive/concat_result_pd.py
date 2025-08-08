import logging
import re
from pathlib import Path
from typing import List

import pandas as pd
from hydra import compose, initialize

from src.analyse.experiment_collection import ExperimentCollection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "config" / "experiments"
RESULT_FILENAME = "results_with_metrics.csv"


def find_experiment_configs() -> List[str]:
    """
    List all config filenames matching the pattern 'config_{model}_{method}.yaml'
    """
    config_files = []
    pattern = re.compile(r"^config_(\w+)_(\w+)\.ya?ml$")

    for file in EXPERIMENT_CONFIG_DIR.iterdir():
        if pattern.match(file.name):
            config_name = file.stem  # without ".yaml"
            config_files.append(config_name)
    return config_files


def extract_model_and_explainer_from_config_name(config_name: str) -> tuple[str, str]:
    """
    From config_resnet50_grad_cam → (resnet50, grad_cam)
    """
    pattern = re.compile(r"^config_(\w+)_(\w+)$")
    match = pattern.match(config_name)
    if not match:
        raise ValueError(f"Invalid config name: {config_name}")
    return match.group(1), match.group(2)


def load_experiment_df(config_name: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    with initialize(config_path="../../config/experiments", version_base=None):
        cfg = compose(config_name=config_name)
        logger.debug(f"Config output dir: {cfg.experiment.output_dir}")

    output_dir = PROJECT_ROOT / Path(cfg.experiment.output_dir)
    csv_path = output_dir / RESULT_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing result CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    model_name, explainer_name = extract_model_and_explainer_from_config_name(
        config_name
    )

    df["model_name"] = model_name
    df["explainer_name"] = explainer_name
    df["experiment_name"] = cfg.experiment.name
    return df


def load_all_experiments(config_names: List[str]) -> ExperimentCollection:
    logger = logging.getLogger(__name__)
    dfs = []
    for cfg_name in config_names:
        try:
            logger.debug(f"Loading experiment data: {cfg_name}")
            df = load_experiment_df(cfg_name)
            dfs.append(df)
            logger.info(f"Loaded experiment: {cfg_name}")
        except Exception as e:
            logger.error(f"Failed to load {cfg_name}: {e}")
    if not dfs:
        raise RuntimeError("No experiments loaded!")
    return ExperimentCollection(dfs)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    config_names = find_experiment_configs()

    if not config_names:
        logger.warning("No matching config files found")
        exit(1)

    exp_collection = load_all_experiments(config_names)

    # 1. Accuracy pro Modell + Erklärmethode
    logger.info("Computing accuracy by model and explainer")
    df_acc = exp_collection.group_accuracy_by_model()
    logger.debug(f"Accuracy results:\n{df_acc}")

    # 2. Korrelationsmatrix aller bekannten Metriken
    logger.info("Computing correlation matrix for all metrics")
    corr = exp_collection.correlation_matrix_all_metrics()
    logger.debug(f"Correlation matrix:\n{corr}")

    # Output-Ordner vorbereiten
    output_dir = PROJECT_ROOT / "results"
    gesamt_analyse_dir = output_dir / "gesamt_analyse"
    gesamt_analyse_dir.mkdir(parents=True, exist_ok=True)

    # 3. Kombinierten DataFrame exportieren
    combined_path = output_dir / "experiments_aggregated.csv"
    exp_collection.df.to_csv(combined_path, index=False)
    logger.info(f"Aggregated CSV saved: {combined_path.resolve()}")

    # 4. Accuracy-CSV speichern
    accuracy_path = gesamt_analyse_dir / "accuracy_by_model_explainer.csv"
    df_acc.to_csv(accuracy_path, index=False)
    logger.info(f"Accuracy CSV saved: {accuracy_path.resolve()}")

    # 5. Korrelationsmatrix als CSV speichern (DataFrame in CSV: numerische Matrix)
    corr_path = gesamt_analyse_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path)
    logger.info(f"Correlation matrix CSV saved: {corr_path.resolve()}")

    plot_dir = gesamt_analyse_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Metriken, die du vergleichen willst
    metrics_to_plot = [
        "iou",
        "prediction_confidence",
        "point_game",
        "pixelprecisionrecall_precision",
        "pixelprecisionrecall_recall",
    ]

    for metric in metrics_to_plot:
        try:
            exp_collection.plot_metric_comparison(metric=metric, save_dir=plot_dir)
            logger.debug(f"Created plot for metric: {metric}")
        except Exception as e:
            logger.warning(f"Failed to create plot for {metric}: {e}")

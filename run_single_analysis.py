#!/usr/bin/env python3
"""
Script zur Durchführung der Single Run Analyse nach Pipeline-Ausführung.

Usage:
    python run_single_analysis.py --config config_resnet50_grad_cam
    python run_single_analysis.py --config config_vgg16_score_cam --output-dir custom_analysis
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import mlflow

from src.pipeline.pipeline_moduls.single_run_analyse.single_run_analysis import SingleRunAnalyse


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def find_experiment_results(config_name: str, project_root: Path) -> Path:
    """
    Findet die Ergebnisse eines Experiments basierend auf dem Config-Namen.
    
    Args:
        config_name: Name der Config (z.B. "config_resnet50_grad_cam")
        project_root: Projekt-Root-Verzeichnis
        
    Returns:
        Pfad zur CSV-Datei mit den Ergebnissen
        
    Raises:
        FileNotFoundError: Wenn keine Ergebnisse gefunden werden
    """
    logger = logging.getLogger(__name__)
    possible_paths = []
    
    # 1. Versuche über Hydra Config das output_dir zu bestimmen
    try:
        from hydra import compose, initialize_config_dir
        config_dir = project_root / "config" / "experiments"
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=f"{config_name}.yaml")
            if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'output_dir'):
                output_dir = Path(cfg.experiment.output_dir)
                if not output_dir.is_absolute():
                    output_dir = project_root / output_dir
                possible_paths.append(output_dir / "results_with_metrics.csv")
                logger.info(f"Config-basierter Pfad: {output_dir}")
    except Exception as e:
        logger.debug(f"Konnte Config {config_name} nicht laden: {e}")
    
    # 2. Fallback: Suche in results-Verzeichnis
    results_dir = project_root / "results"
    config_base = config_name.replace("config_", "")
    
    # Mögliche Ordnernamen basierend auf Config
    possible_patterns = [
        f"*{config_base}*",
        f"*{config_base.split('_')[0]}*",  # nur Modellname
        "experiment_*",
        "*_experiment",
    ]
    
    for pattern in possible_patterns:
        for exp_dir in results_dir.glob(pattern):
            if exp_dir.is_dir():
                csv_file = exp_dir / "results_with_metrics.csv"
                if csv_file.exists():
                    possible_paths.append(csv_file)
    
    # 3. Durchsuche alle Experiment-Ordner
    if results_dir.exists():
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                csv_file = exp_dir / "results_with_metrics.csv"
                if csv_file.exists():
                    possible_paths.append(csv_file)
    
    # Versuche die Pfade und prüfe ob sie zur Config passen
    for csv_path in possible_paths:
        if csv_path.exists():
            try:
                # Prüfe ob die Datei lesbar ist und Daten enthält
                df = pd.read_csv(csv_path, nrows=5)
                if len(df) > 0:
                    logger.info(f"Gefundene Ergebnisse: {csv_path}")
                    logger.info(f"Samples: {len(pd.read_csv(csv_path))}")
                    return csv_path
            except Exception as e:
                logger.warning(f"Konnte {csv_path} nicht lesen: {e}")
                continue
    
    # Zeige verfügbare Dateien falls nichts gefunden wurde
    logger.error(f"Keine Ergebnisse für Config '{config_name}' gefunden!")
    logger.info("Verfügbare Experiment-Ergebnisse:")
    if results_dir.exists():
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                csv_file = exp_dir / "results_with_metrics.csv"
                if csv_file.exists():
                    logger.info(f"  - {exp_dir.name}: {csv_file}")
    
    raise FileNotFoundError(f"Keine Ergebnisse für Config '{config_name}' gefunden")


def run_single_analysis(config_name: str, output_dir: Path = None, mlflow_logging: bool = True) -> Path:
    """
    Führt die Single Run Analyse für eine gegebene Config durch.
    
    Args:
        config_name: Name der Config (z.B. "config_resnet50_grad_cam")
        output_dir: Optionales benutzerdefiniertes Ausgabeverzeichnis
        mlflow_logging: Ob Ergebnisse zu MLflow geloggt werden sollen
        
    Returns:
        Pfad zum Ausgabeverzeichnis der Analyse
    """
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).resolve().parent
    
    # 1. Finde Experiment-Ergebnisse
    logger.info(f"Suche Ergebnisse für Config: {config_name}")
    csv_path = find_experiment_results(config_name, project_root)
    
    # 2. Lade Daten
    logger.info(f"Lade Daten aus: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Geladene Daten: {len(df)} Samples")
    
    # 3. Bestimme Ausgabeverzeichnis
    if output_dir is None:
        # Standard: neben den Original-Ergebnissen
        output_dir = csv_path.parent / "single_run_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ausgabeverzeichnis: {output_dir}")
    
    # 4. Führe Single Run Analyse durch
    logger.info("Starte Single Run Analyse...")
    analysis = SingleRunAnalyse(df)
    
    all_plots = {}
    
    # 4.1 IoU Histogramme
    logger.info("Erstelle IoU Histogramme...")
    try:
        iou_plots = analysis.plot_iou_histograms_by_correctness(data_dir)
        all_plots.update(iou_plots)
        logger.info(f"IoU Histogramme erstellt: {len(iou_plots)}")
    except Exception as e:
        logger.warning(f"IoU Histogramme fehlgeschlagen: {e}")
    
    # 4.2 Andere Metriken Histogramme
    logger.info("Erstelle Prediction Correctness Histogramme...")
    try:
        other_plots = analysis.plot_prediction_correctness_histograms(data_dir)
        all_plots.update(other_plots)
        logger.info(f"Prediction Correctness Histogramme erstellt: {len(other_plots)}")
    except Exception as e:
        logger.warning(f"Prediction Correctness Histogramme fehlgeschlagen: {e}")
    
    # 4.3 Pixel Precision Histogramme
    logger.info("Erstelle Pixel Precision Histogramme...")
    try:
        pixel_prec_plots = analysis.plot_pixel_precision_histograms_by_correctness(data_dir)
        all_plots.update(pixel_prec_plots)
        logger.info(f"Pixel Precision Histogramme erstellt: {len(pixel_prec_plots)}")
    except Exception as e:
        logger.warning(f"Pixel Precision Histogramme fehlgeschlagen: {e}")
    
    # 4.4 Pixel Recall Histogramme
    logger.info("Erstelle Pixel Recall Histogramme...")
    try:
        pixel_rec_plots = analysis.plot_pixel_recall_histograms_by_correctness(data_dir)
        all_plots.update(pixel_rec_plots)
        logger.info(f"Pixel Recall Histogramme erstellt: {len(pixel_rec_plots)}")
    except Exception as e:
        logger.warning(f"Pixel Recall Histogramme fehlgeschlagen: {e}")
    
    # 4.5 Korrelationen berechnen
    logger.info("Berechne Korrelationen...")
    try:
        correlations = analysis.correlation_with_correctness()
        corr_df = pd.DataFrame(list(correlations.items()), 
                              columns=['metric', 'correlation_with_correctness'])
        corr_csv_path = data_dir / "correlations.csv"
        corr_df.to_csv(corr_csv_path, index=False)
        logger.info(f"Korrelationen gespeichert: {corr_csv_path}")
    except Exception as e:
        logger.warning(f"Korrelationen fehlgeschlagen: {e}")
    
    # 4.6 Statistische Tests berechnen
    logger.info("Berechne statistische Tests...")
    try:
        statistical_tests_df = analysis.calculate_statistical_tests_for_all_metrics()
        if not statistical_tests_df.empty:
            stats_csv_path = data_dir / "statistical_tests.csv"
            statistical_tests_df.to_csv(stats_csv_path, index=False)
            logger.info(f"Statistische Tests gespeichert: {stats_csv_path}")
            logger.info(f"Tests für {len(statistical_tests_df)} Metriken durchgeführt")
        else:
            logger.warning("Keine Metriken für statistische Tests verfügbar")
    except Exception as e:
        logger.warning(f"Statistische Tests fehlgeschlagen: {e}")
    
    # 5. MLflow Logging (optional)
    if mlflow_logging:
        try:
            logger.info("Logge Ergebnisse zu MLflow...")
            
            # Starte oder nutze existierenden MLflow Run
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=f"single_analysis_{config_name}")
            
            # Logge alle Plots
            for plot_name, plot_path in all_plots.items():
                mlflow.log_artifact(str(plot_path), artifact_path="single_analysis/plots")
            
            # Logge CSV Dateien
            for data_file in data_dir.glob("*.csv"):
                mlflow.log_artifact(str(data_file), artifact_path="single_analysis/data")
            
            # Logge Parameter
            mlflow.log_param("config_name", config_name)
            mlflow.log_param("samples_analyzed", len(df))
            mlflow.log_param("plots_created", len(all_plots))
            
            logger.info("MLflow Logging abgeschlossen")
            
        except Exception as e:
            logger.warning(f"MLflow Logging fehlgeschlagen: {e}")
    
    logger.info(f"Single Run Analyse abgeschlossen!")
    logger.info(f"Ergebnisse in: {output_dir}")
    logger.info(f"Plots erstellt: {len(all_plots)}")
    
    return output_dir


def main():
    """Hauptfunktion mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Führt Single Run Analyse nach Pipeline-Ausführung durch"
    )
    parser.add_argument(
        "--config", 
        required=True,
        help="Name der Config (z.B. 'config_resnet50_grad_cam')"
    )
    parser.add_argument(
        "--output-dir",
        help="Benutzerdefiniertes Ausgabeverzeichnis (optional)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Deaktiviert MLflow Logging"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Aktiviert ausführliche Ausgabe (DEBUG Level)"
    )
    
    args = parser.parse_args()
    
    # Setup Logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = setup_logging()
    
    try:
        # Führe Analyse durch
        output_dir = run_single_analysis(
            config_name=args.config,
            output_dir=args.output_dir,
            mlflow_logging=not args.no_mlflow
        )
        
        print(f"\n[SUCCESS] Single Run Analyse erfolgreich abgeschlossen!")
        print(f"[RESULTS] Ergebnisse: {output_dir}")
        print(f"[PLOTS] Plots: {output_dir / 'plots'}")
        print(f"[DATA] Daten: {output_dir / 'data'}")
        
    except FileNotFoundError as e:
        logger.error(f"Fehler: {e}")
        print(f"\n[ERROR] Fehler: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        print(f"\n[ERROR] Unerwarteter Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
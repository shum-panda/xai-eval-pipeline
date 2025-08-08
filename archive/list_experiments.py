#!/usr/bin/env python3
"""
Helper-Script zur Auflistung aller verfügbaren Experimente.

Usage:
    python list_experiments.py
    python list_experiments.py --detailed
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime


def list_available_experiments(detailed: bool = False) -> None:
    """
    Listet alle verfügbaren Experimente auf.
    
    Args:
        detailed: Wenn True, zeigt detaillierte Informationen über jedes Experiment
    """
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    
    if not results_dir.exists():
        print("[ERROR] Results-Verzeichnis existiert nicht!")
        return
    
    print("=" * 80)
    print("VERFÜGBARE EXPERIMENTE")
    print("=" * 80)
    
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            csv_file = exp_dir / "results_with_metrics.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, nrows=5)
                    full_df = pd.read_csv(csv_file)
                    
                    # Sammle Grundinformationen
                    experiment = {
                        'directory': exp_dir.name,
                        'csv_path': csv_file,
                        'samples': len(full_df),
                        'columns': list(df.columns),
                        'model_name': full_df['model_name'].iloc[0] if 'model_name' in full_df.columns else 'Unknown',
                        'explainer_name': full_df['explainer_name'].iloc[0] if 'explainer_name' in full_df.columns else 'Unknown',
                        'file_size': csv_file.stat().st_size / (1024*1024),  # MB
                        'modified': datetime.fromtimestamp(csv_file.stat().st_mtime)
                    }
                    
                    if detailed:
                        # Zusätzliche Details für detaillierte Ansicht
                        experiment['accuracy'] = full_df['prediction_correct'].mean() if 'prediction_correct' in full_df.columns else None
                        experiment['avg_iou'] = full_df['iou'].mean() if 'iou' in full_df.columns else None
                        experiment['avg_point_game'] = full_df['point_game'].mean() if 'point_game' in full_df.columns else None
                    
                    experiments.append(experiment)
                    
                except Exception as e:
                    print(f"[WARNING] Fehler beim Lesen von {csv_file.name}: {e}")
    
    if not experiments:
        print("[ERROR] Keine Experimente gefunden!")
        return
    
    # Sortiere nach Änderungsdatum (neueste zuerst)
    experiments.sort(key=lambda x: x['modified'], reverse=True)
    
    print(f"[INFO] {len(experiments)} Experimente gefunden:\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"[{i:2d}] [FOLDER] {exp['directory']}")
        print(f"     [MODEL] Model: {exp['model_name']}")
        print(f"     [EXPLAINER] Explainer: {exp['explainer_name']}")
        print(f"     [SAMPLES] Samples: {exp['samples']:,}")
        print(f"     [SIZE] Dateigröße: {exp['file_size']:.1f} MB")
        print(f"     [TIME] Geändert: {exp['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if detailed:
            if exp['accuracy'] is not None:
                print(f"     [ACC] Genauigkeit: {exp['accuracy']:.3f}")
            if exp['avg_iou'] is not None:
                print(f"     [IOU] Durchschn. IoU: {exp['avg_iou']:.3f}")
            if exp['avg_point_game'] is not None:
                print(f"     [GAME] Durchschn. Point Game: {exp['avg_point_game']:.3f}")
            print(f"     [COLS] Spalten ({len(exp['columns'])}): {', '.join(exp['columns'][:5])}{'...' if len(exp['columns']) > 5 else ''}")
        
        print()
    
    print("=" * 80)
    print("VERWENDUNG:")
    print("Führe Single Run Analyse aus mit:")
    print("  python run_single_analysis.py --config <config_name>")
    print("\nBeispiele:")
    for exp in experiments[:3]:  # Zeige Top 3 Beispiele
        # Versuche Config-Namen zu rekonstruieren
        model = exp['model_name'].lower()
        explainer = exp['explainer_name'].lower().replace(' ', '_')
        config_name = f"config_{model}_{explainer}"
        print(f"  python run_single_analysis.py --config {config_name}")
    print("=" * 80)


def main():
    """Hauptfunktion mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Listet alle verfügbaren Experimente auf"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Zeigt detaillierte Informationen über jedes Experiment"
    )
    
    args = parser.parse_args()
    
    list_available_experiments(detailed=args.detailed)


if __name__ == "__main__":
    main()
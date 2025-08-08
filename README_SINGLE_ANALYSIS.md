# Single Run Analysis Scripts

Diese Scripts ermöglichen es, nach dem Ende einer Pipeline-Ausführung eine detaillierte Single Run Analyse durchzuführen.

## 📋 Übersicht

### Verfügbare Scripts:

1. **`run_single_analysis.py`** - Führt die Single Run Analyse für ein spezifisches Experiment durch
2. **`list_experiments.py`** - Listet alle verfügbaren Experimente auf
3. **`diagnose_data.py`** - Diagnostiziert verfügbare Experiment-Daten (bereits vorhanden)

## 🚀 Schnellstart

### 1. Verfügbare Experimente anzeigen
```bash
# Einfache Auflistung
python list_experiments.py

# Detaillierte Auflistung mit Metriken
python list_experiments.py --detailed
```

### 2. Single Run Analyse durchführen
```bash
# Grundlegende Analyse
python run_single_analysis.py --config config_resnet50_grad_cam

# Mit benutzerdefiniertem Ausgabeverzeichnis
python run_single_analysis.py --config config_vgg16_score_cam --output-dir my_analysis

# Ohne MLflow Logging
python run_single_analysis.py --config config_resnet50_integrated_gradients --no-mlflow

# Mit ausführlicher Ausgabe
python run_single_analysis.py --config config_resnet18_grad_cam --verbose
```

## 📊 Was wird erstellt?

Die Single Run Analyse erstellt folgende Visualisierungen und Daten:

### Histogramme:
- **IoU Histogramme** - aufgeteilt nach Prediction Correctness
- **Prediction Correctness Histogramme** - für verschiedene Metriken
- **Pixel Precision Histogramme** - aufgeteilt nach Prediction Correctness
- **Pixel Recall Histogramme** - aufgeteilt nach Prediction Correctness

### Daten:
- **CSV-Dateien** - Histogram-Daten für jede Analyse
- **Korrelationen** - Korrelation zwischen Metriken und Prediction Correctness

### Ausgabestruktur:
```
output_directory/
├── plots/                          # Alle generierten Plots
│   ├── iou_histogram_correct.png
│   ├── iou_histogram_incorrect.png
│   ├── pixel_precision_histogram_correct.png
│   └── ...
├── data/                           # CSV-Daten und Korrelationen
│   ├── iou_histogram_data.csv
│   ├── pixel_precision_histogram_data.csv
│   ├── correlations.csv
│   └── ...
```

## 🔧 Erweiterte Optionen

### Automatische Pfad-Erkennung
Das Script erkennt automatisch die Experiment-Ergebnisse basierend auf:
1. **Hydra Config** - Lädt die Config und verwendet das definierte `output_dir`
2. **Pattern Matching** - Sucht in `results/` nach Ordnern, die zum Config-Namen passen
3. **Fallback** - Durchsucht alle Experiment-Ordner nach `results_with_metrics.csv`

### MLflow Integration
- Standardmäßig werden alle Ergebnisse zu MLflow geloggt
- Plots werden unter `single_analysis/plots` gespeichert
- CSV-Daten unter `single_analysis/data`
- Mit `--no-mlflow` kann das Logging deaktiviert werden

### Logging
- Standardmäßig INFO-Level Logging
- Mit `--verbose` wird DEBUG-Level aktiviert
- Alle wichtigen Schritte werden protokolliert

## 📝 Beispiele

### Beispiel 1: ResNet50 mit GradCAM analysieren
```bash
# Zuerst verfügbare Experimente auflisten
python list_experiments.py

# Single Run Analyse durchführen
python run_single_analysis.py --config config_resnet50_grad_cam

# Ergebnis: Analyse wird in results/resnet50_grad_cam/single_run_analysis/ gespeichert
```

### Beispiel 2: Benutzerdefinierte Analyse
```bash
# Analyse mit eigenem Ausgabeverzeichnis
python run_single_analysis.py \
    --config config_vgg16_score_cam \
    --output-dir ./custom_analysis_vgg16 \
    --verbose

# Ergebnis: Analyse wird in ./custom_analysis_vgg16/ gespeichert
```

### Beispiel 3: Batch-Analyse mehrerer Experimente
```bash
# Alle ResNet Experimente analysieren
python run_single_analysis.py --config config_resnet18_grad_cam
python run_single_analysis.py --config config_resnet50_grad_cam
python run_single_analysis.py --config config_resnet50_guided_backprop
python run_single_analysis.py --config config_resnet50_integrated_gradients
```

## ⚠️ Hinweise

### Voraussetzungen:
- Das Experiment muss erfolgreich abgeschlossen sein
- Eine `results_with_metrics.csv` Datei muss existieren
- Die CSV muss die erforderlichen Spalten enthalten (iou, point_game, prediction_correct, etc.)

### Häufige Probleme:
1. **Config nicht gefunden**: Überprüfe den Config-Namen mit `list_experiments.py`
2. **CSV nicht lesbar**: Stelle sicher, dass das Experiment vollständig abgeschlossen ist
3. **MLflow Fehler**: Verwende `--no-mlflow` falls MLflow nicht konfiguriert ist

### Performance:
- Die Analyse läuft typischerweise sehr schnell (< 1 Minute)
- Größere Datensätze (>10k Samples) können etwas länger dauern
- Alle Plots werden in hoher Qualität (300 DPI) gespeichert

## 🔄 Integration mit der Pipeline

### Nach Pipeline-Ausführung:
```bash
# 1. Pipeline ausführen
python main.py experiment=config_resnet50_grad_cam

# 2. Single Run Analyse durchführen
python run_single_analysis.py --config config_resnet50_grad_cam

# 3. Für erweiterte Multi-Experiment Analyse (optional)
python src/analyse/simple_analyzer.py
```

### Automatisierung:
Das Script kann leicht in Bash/PowerShell Scripts integriert werden:

```bash
#!/bin/bash
CONFIGS=("config_resnet50_grad_cam" "config_vgg16_score_cam" "config_resnet18_grad_cam")

for config in "${CONFIGS[@]}"; do
    echo "Analysiere $config..."
    python run_single_analysis.py --config "$config"
done
```

## 📚 Weitere Ressourcen

- **Diagnose-Script**: `python diagnose_data.py` - Überprüft verfügbare Daten
- **Simple Analyzer**: `python src/analyse/simple_analyzer.py` - Erweiterte Multi-Experiment Analyse
- **Pipeline Dokumentation**: Siehe Haupt-README für Pipeline-Nutzung
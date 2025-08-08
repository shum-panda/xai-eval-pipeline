# XAI Evaluation Pipeline - Konfigurationsdateien Dokumentation

Diese umfangreiche Dokumentation erklärt alle verfügbaren Konfigurationsoptionen für die XAI Evaluation Pipeline. Die Pipeline verwendet YAML-Konfigurationsdateien zur Steuerung von Experimenten.

## 📋 Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Konfigurationsstruktur](#konfigurationsstruktur)
3. [Detaillierte Konfigurationssektionen](#detaillierte-konfigurationssektionen)
4. [⚠️ Wichtige Hinweise und Bekannte Probleme](#️-wichtige-hinweise-und-bekannte-probleme)
5. [Beispielkonfigurationen](#beispielkonfigurationen)
6. [Validierung und Fehlerbehandlung](#validierung-und-fehlerbehandlung)
7. [Best Practices](#best-practices)

## Überblick

Die XAI Evaluation Pipeline verwendet **Hydra** mit **OmegaConf** zur Konfigurationsverwaltung. Alle Konfigurationsdateien befinden sich im Verzeichnis `config/experiments/` und folgen einer hierarchischen Struktur.

### Verfügbare Konfigurationsdateien

```
config/experiments/
├── config_resnet18_grad_cam.yaml
├── config_resnet34_grad_cam.yaml
├── config_resnet50_grad_cam.yaml
├── config_resnet50_guided_backprop.yaml
├── config_resnet50_integrated_gradients.yaml
├── config_resnet50_score_cam.yaml
├── config_vgg16_grad_cam.yaml
├── config_vgg16_score_cam.yaml
├── config_example_mlflow_setup.yaml
├── config_grad_cam_no_limit.yaml
└── config_test_quick.yaml
```

## Konfigurationsstruktur

Jede Konfigurationsdatei besteht aus **9 Hauptsektionen**:

```yaml
experiment:    # Grundlegende Experiment-Einstellungen
hardware:      # Hardware-Konfiguration (⚠️ teilweise deprecated)
model:         # Modell-Konfiguration
data:          # Daten- und Datensatz-Einstellungen
xai:           # XAI-Methoden-Konfiguration
metric:        # Evaluationsmetriken
visualization: # Visualisierungsoptionen
logging:       # Logging-Einstellungen
mlflow:        # MLflow-Integration
```

## Detaillierte Konfigurationssektionen

### 1. 📊 Experiment-Sektion

Grundlegende Informationen über das Experiment.

```yaml
experiment:
  name: "resnet50_grad_cam"           # Name des Experiments (für Logging)
  output_dir: "results/resnet50_grad_cam"  # Ausgabeverzeichnis für Ergebnisse
  top_k: 10                           # Top-K Vorhersagen berücksichtigen
  seed: 42                            # Zufallsseed für Reproduzierbarkeit
```

**Parameter:**
- `name` (str): Eindeutiger Name für das Experiment
- `output_dir` (str): Pfad zum Speichern der Ergebnisse (relativ oder absolut)
- `top_k` (int): Anzahl der Top-Vorhersagen für die Evaluation
- `seed` (int): Zufallsseed für reproduzierbare Ergebnisse

### 2. 🖥️ Hardware-Sektion ⚠️

**⚠️ WICHTIGER HINWEIS:** Diese Sektion enthält **deprecated/nicht-funktionale** Parameter!

```yaml
hardware:
  use_cuda: true      # ❌ HAT KEINE WIRKUNG - wird ignoriert
  device: "cuda:0"    # ❌ WIRD KOMPLETT IGNORIERT
```

**❌ Bekannte Probleme:**
- `device`: Wird **vollständig ignoriert**. Das System erkennt automatisch verfügbare Hardware
- `use_cuda`: Wird nur für Logging verwendet, aber die tatsächliche Gerätewahl erfolgt automatisch
- **Automatische Geräteerkennung:** Das System verwendet `torch.cuda.is_available()` zur Geräteauswahl

**Tatsächliches Verhalten:**
```python
# Code aus src/pipeline/pipeline_moduls/models/base/xai_model.py
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3. 🤖 Model-Sektion

Konfiguration des neuronalen Netzwerkmodells.

```yaml
model:
  name: "resnet50"        # Modellarchitektur
  pretrained: true        # Vortrainierte Gewichte verwenden
  weights_path: null      # Pfad zu benutzerdefinierten Gewichten (optional)
  transform: true         # Modellspezifische Transformationen anwenden
```

**Unterstützte Modelle:**
- `resnet18`, `resnet34`, `resnet50`
- `vgg16`
- Weitere können durch Erweiterung der `ModelFactory` hinzugefügt werden

**Parameter:**
- `name` (str): Modellarchitektur-Name
- `pretrained` (bool): ImageNet-vortrainierte Gewichte laden
- `weights_path` (str|null): Pfad zu benutzerdefinierten Gewichtsdateien
- `transform` (bool): Standard-Bildtransformationen aktivieren

### 4. 📁 Data-Sektion

Umfassende Datensatz- und Datenverarbeitungsoptionen.

```yaml
data:
  dataset_name: "imagenet_val"                    # Name des Datensatzes
  dataset_path: "data/extracted/validation_images"  # Pfad zu Bildern
  annotation_path: "data/extracted/bounding_boxes"  # Pfad zu Bounding Boxes
  label_file: "data/ILSVRC2012_validation_ground_truth.txt"  # Ground Truth Labels
  
  # Datenverarbeitung
  shuffle: false              # Datenreihenfolge mischen
  resize: [224, 224]          # Bildgröße (Breite, Höhe) - VALIDIERT!
  normalize: true             # Bildnormalisierung anwenden
  
  # Datenaugmentation
  augmentation:
    horizontal_flip: false    # Zufällige horizontale Spiegelungen
    random_crop: false        # Zufällige Bildausschnitte
  
  # Batch-Verarbeitung
  batch_size: 32              # Anzahl Samples pro Batch
  max_batches: 300            # Maximale Anzahl Batches (null = unbegrenzt)
  num_workers: 4              # Anzahl Worker für Datenladung
  pin_memory: true            # Pinned Memory für CUDA-Beschleunigung
```

**✅ Validierung:**
- `resize`: Muss eine Liste mit exakt 2 Ganzzahlen sein
- Pfade werden zur Laufzeit validiert

**Performance-Tipps:**
- `num_workers`: 4-8 für SSDs, 2-4 für HDDs
- `pin_memory: true` bei CUDA-Nutzung für bessere Performance
- `batch_size`: Abhängig von GPU-Speicher (8-32 typisch)

### 5. 🔍 XAI-Sektion

Konfiguration der Explainable AI Methoden.

```yaml
xai:
  name: "grad_cam"        # XAI-Methode
  use_defaults: true      # Standard-Parameter verwenden
  kwargs: {}              # Methodenspezifische Parameter
```

**Verfügbare XAI-Methoden:**
- `grad_cam`: Gradient-weighted Class Activation Mapping
- `score_cam`: Score-weighted Class Activation Mapping  
- `guided_backprop`: Guided Backpropagation
- `integrated_gradients`: Integrated Gradients

**Methodenspezifische Parameter:**

#### GradCAM-Konfiguration:
```yaml
xai:
  name: "grad_cam"
  use_defaults: false
  kwargs:
    target_layer: -1           # Zielschicht (Index oder Name)
    relu_attributions: true    # ReLU auf Attributionen anwenden
    interpolate_mode: "bilinear"  # Interpolationsmodus
```

#### ScoreCAM-Konfiguration:
```yaml
xai:
  name: "score_cam" 
  use_defaults: false
  kwargs:
    target_layer: -1           # Zielschicht
    batch_size: 32            # Batch-Größe für Score-Berechnung
```

**Validierung:**
- `target_layer`: Muss int oder str sein
- `interpolate_mode`: Validiert gegen erlaubte Modi ("bilinear", "nearest", etc.)

### 6. 📏 Metric-Sektion

Konfiguration der Evaluationsmetriken.

```yaml
metric:
  kwargs:
    iou: 
      threshold: 0.3                    # IoU-Schwellenwert
    pixel_precision_recall:
      threshold: 0.3                    # Pixel-basierte Metriken-Schwellenwert
```

**Verfügbare Metriken:**
- `iou`: Intersection over Union
- `pixel_precision_recall`: Pixel-basierte Präzision und Recall
- `point_game`: Point Game Metric (automatisch aktiv)

**Parameter:**
- `threshold` (float): Schwellenwert für binäre Klassifikation (0.0-1.0)

### 7. 🎨 Visualization-Sektion

Konfiguration der Visualisierungsausgabe.

```yaml
visualization:
  save: true                    # Visualisierungen speichern
  show: false                   # Visualisierungen anzeigen (nicht in Batch-Jobs)
  max_visualizations: 50        # Maximale Anzahl gespeicherter Visualisierungen
```

**⚠️ Bekannte Einschränkungen:**
- **DPI ist fest auf 150 kodiert** in der Implementierung
- **Farbschemas, Overlay-Modi und Dateiformate sind hardcoded**
- `show: true` funktioniert nicht in SSH/Docker-Umgebungen ohne Display

**Hardcoded-Werte:**
- DPI: 150
- Format: PNG
- Colormap: 'jet'
- Facecolor: 'white'

### 8. 📝 Logging-Sektion

Logging-Konfiguration für Debug und Monitoring.

```yaml
logging:
  level: "INFO"               # Logging-Level
```

**Verfügbare Logging-Level:**
- `DEBUG`: Sehr detaillierte Informationen
- `INFO`: Allgemeine Informationen (empfohlen für normale Nutzung)
- `WARNING`: Nur Warnungen und Fehler
- `ERROR`: Nur Fehler

### 9. 📊 MLflow-Sektion

Umfangreiche MLflow-Integration für Experiment-Tracking.

```yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"       # MLflow-Tracking-URI
  experiment_name: "xai_evaluation"         # Experiment-Name in MLflow
  run_name: "ResNet50_GradCAM_Run"         # Benutzerdefinierter Run-Name
  auto_log: true                           # Automatisches Logging aktivieren
  artifact_location: null                  # Benutzerdefinierter Artifact-Speicher
  
  tags:                                    # Benutzerdefinierte Tags
    model_family: "resnet"
    xai_method: "grad_cam"
    dataset: "imagenet_val"
    purpose: "baseline"
    batch_size: "32"
    framework: "pytorch"
    project: "xai-eval-pipeline"
    environment: "development"
    author: "your-name"
    description: "Experiment-Beschreibung"
```

**MLflow-URI-Optionen:**
- `sqlite:///mlflow.db`: Lokale SQLite-Datenbank
- `http://localhost:5000`: Remote MLflow-Server
- `file:///path/to/mlruns`: Lokales Dateisystem

**Automatisch geloggte Informationen:**
- Modellparameter und -architektur
- Datensatz-Informationen
- XAI-Methoden-Parameter
- Evaluation-Metriken
- Konfigurationsdateien als Artifacts
- System-Informationen

## ⚠️ Wichtige Hinweise und Bekannte Probleme

### ❌ Nicht-funktionale Parameter

1. **`hardware.device`**: Wird **komplett ignoriert**
   - Das System verwendet automatische Geräteerkennung
   - Konfiguration hat keine Wirkung auf tatsächliche Geräteauswahl

2. **`hardware.use_cuda`**: Hat **keine funktionale Wirkung**
   - Wird nur für MLflow-Logging verwendet
   - Tatsächliche CUDA-Nutzung wird automatisch erkannt

### ⚠️ Eingeschränkte Funktionalität

1. **Visualization-Parameter sind hardcoded:**
   - DPI: 150 (nicht konfigurierbar)
   - Colormap: 'jet' (nicht konfigurierbar)
   - Dateiformat: PNG (nicht konfigurierbar)

2. **Device-Selection ist automatisch:**
   - Verwendet `torch.cuda.is_available()`
   - Keine manuelle Kontrolle möglich

### 🔧 Workarounds

Für erweiterte Anpassungen:
1. **Hardware-Kontrolle**: Code-Modifikation in `XAIModel` erforderlich
2. **Visualisierung**: Anpassung in `visualization.py` nötig
3. **Batch-Größe für GPU**: Reduzieren Sie `batch_size` bei Out-of-Memory-Fehlern

## Beispielkonfigurationen

### Schneller Test-Lauf

```yaml
# config_test_quick.yaml - Für schnelle Entwicklungstests
experiment:
  name: "test_quick_run"
  output_dir: "results/test_quick_run"
  seed: 42

hardware:
  use_cuda: true      # Wird ignoriert, aber für Konsistenz beibehalten
  device: "cuda:0"    # Wird ignoriert

model:
  name: "resnet18"    # Kleineres Modell für Geschwindigkeit
  pretrained: true

data:
  dataset_path: "data/extracted/validation_images"
  batch_size: 8       # Kleine Batch-Größe
  max_batches: 5      # Nur 5 Batches verarbeiten
  num_workers: 2

xai:
  name: "grad_cam"    # Schnelle XAI-Methode

visualization:
  max_visualizations: 5  # Wenige Visualisierungen

logging:
  level: "DEBUG"      # Detailliertes Logging für Tests
```

### Produktions-Setup

```yaml
# config_production.yaml - Für vollständige Evaluationen
experiment:
  name: "production_resnet50_eval"
  output_dir: "results/production_run"
  seed: 42

model:
  name: "resnet50"
  pretrained: true

data:
  dataset_path: "data/extracted/validation_images"
  batch_size: 32
  max_batches: null    # Alle Daten verarbeiten
  num_workers: 8       # Mehr Workers für Performance

xai:
  name: "grad_cam"
  use_defaults: false
  kwargs:
    target_layer: -2   # Spezifische Schicht
    relu_attributions: true

metric:
  kwargs:
    iou: { threshold: 0.5 }  # Höherer Schwellenwert
    pixel_precision_recall: { threshold: 0.5 }

visualization:
  save: true
  max_visualizations: 200

mlflow:
  tracking_uri: "http://mlflow-server:5000"  # Remote-Server
  experiment_name: "production_xai_evaluation"
  tags:
    environment: "production"
    dataset_version: "v2.0"
    model_version: "resnet50_v1"
```

### MLflow-Demo-Setup

```yaml
# config_mlflow_demo.yaml - MLflow-Integration demonstrieren
experiment:
  name: "mlflow_integration_demo"
  output_dir: "results/mlflow_demo"

data:
  batch_size: 8
  max_batches: 10      # Begrenzt für Demo

mlflow:
  tracking_uri: "sqlite:///mlflow_demo.db"
  experiment_name: "xai_demo_experiment"
  run_name: "demo_run_with_comprehensive_tags"
  auto_log: true
  tags:
    # Detaillierte Tags für bessere Organisation
    model_family: "resnet"
    xai_method: "grad_cam" 
    dataset: "imagenet_val"
    purpose: "demo"
    batch_size: "8"
    framework: "pytorch"
    project: "xai-eval-pipeline"
    environment: "development"
    author: "data-scientist"
    description: "Demonstration der MLflow-Integration"
    experiment_type: "xai_evaluation"
    data_version: "v1.0"
```

## Validierung und Fehlerbehandlung

### Automatische Validierungen

1. **DataConfig.resize**: Muss Liste mit 2 Ganzzahlen sein
2. **XAI-Parameter**: Methodenspezifische Validierung
3. **Pfad-Existenz**: Wird zur Laufzeit geprüft

### Häufige Fehler und Lösungen

| Fehler | Ursache | Lösung |
|--------|---------|---------|
| `CUDA out of memory` | `batch_size` zu groß | Reduzieren Sie `batch_size` auf 8-16 |
| `FileNotFoundError` | Datenpfade falsch | Überprüfen Sie `dataset_path`, `annotation_path` |
| `ModuleNotFoundError` | XAI-Methode nicht implementiert | Verwenden Sie unterstützte Methoden |
| `ValueError: resize must be list` | Falsches `resize`-Format | Verwenden Sie `[224, 224]` Format |

### Debug-Strategien

```yaml
# Für Debugging
logging:
  level: "DEBUG"       # Detaillierte Logs

data:
  batch_size: 4        # Kleine Batches
  max_batches: 2       # Wenige Batches
  num_workers: 0       # Kein Multiprocessing

visualization:
  max_visualizations: 3  # Wenige Visualisierungen
```

## Best Practices

### 1. 🏃‍♂️ Performance-Optimierung

```yaml
# Für beste Performance
data:
  batch_size: 32       # Abhängig von GPU-Memory
  num_workers: 8       # 2 × CPU-Kerne
  pin_memory: true     # Bei CUDA-Nutzung
  max_batches: null    # Für vollständige Evaluation

hardware:
  use_cuda: true       # Auch wenn ignoriert, für MLflow-Logging
```

### 2. 🧪 Experimentorganisation

```yaml
# Strukturierte Experiment-Namen
experiment:
  name: "{model}_{xai_method}_{dataset}_{version}"
  output_dir: "results/{experiment.name}"

mlflow:
  experiment_name: "xai_evaluation_{project_phase}"
  tags:
    project: "my_xai_research"
    phase: "baseline|optimization|final"
    version: "v1.0"
```

### 3. 📊 Monitoring und Tracking

```yaml
# Umfassendes MLflow-Setup
mlflow:
  auto_log: true
  tags:
    # Technische Details
    model_family: "{model_type}"
    xai_method: "{xai_name}"
    framework: "pytorch"
    
    # Experiment-Context
    purpose: "baseline|comparison|optimization"
    environment: "dev|test|prod"
    dataset_version: "v{X.Y}"
    
    # Metadata
    author: "{your_name}"
    project: "{project_name}"
    description: "Kurze Beschreibung des Experiments"
```

### 4. 🔄 Reproduzierbarkeit

```yaml
# Für reproduzierbare Ergebnisse
experiment:
  seed: 42             # Fester Seed

data:
  shuffle: false       # Konsistente Reihenfolge
  
# Dokumentation in MLflow-Tags
mlflow:
  tags:
    config_hash: "{hash_of_config}"
    git_commit: "{git_commit_hash}"
    timestamp: "{iso_timestamp}"
```

### 5. 🎯 Batch-Processing

```yaml
# Für automatisierte Batch-Läufe
visualization:
  show: false          # Kein GUI in Batch-Jobs
  save: true
  max_visualizations: 100

logging:
  level: "INFO"        # Nicht zu verbose für Batch-Jobs

data:
  num_workers: 8       # Maximale Parallelität
```

---

## 📞 Support und Weiterentwicklung

Bei Problemen oder Feature-Requests:

1. **Überprüfen Sie diese Dokumentation** für bekannte Probleme
2. **Aktivieren Sie Debug-Logging** für detaillierte Informationen
3. **Testen Sie mit `config_test_quick.yaml`** für schnelle Validierung
4. **Prüfen Sie MLflow-Logs** für detaillierte Experiment-Informationen

### Erweiterungen

Um neue Funktionen hinzuzufügen:
- Neue Modelle: Erweitern Sie `ModelFactory`
- Neue XAI-Methoden: Implementieren Sie `BaseXAIConfig` und `BaseXAI`
- Neue Metriken: Erweitern Sie `MetricConfig` und Evaluator

---

*Letzte Aktualisierung: August 2025*
*Version: 1.0*
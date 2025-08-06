# 📊 XAI Experiment Analysis Suite

Erweiterte Analyse-Suite für XAI-Experimente mit umfangreichen Visualisierungen und Datenauswertungen.

## 🎯 **Übersicht**

Das neue Analyse-System bietet:
- **Separate Analysen** für verschiedene Experiment-Typen
- **20+ verschiedene Plot-Typen** für tiefgreifende Datenanalyse
- **Konfigurierbare Experiment-Listen** über YAML
- **Erweiterte statistische Auswertungen**

## 📁 **Datei-Struktur**

```
src/analyse/
├── experiment_analyzer.py          # Basis-Analyzer
├── config_based_analyzer.py        # YAML-konfigurierbarer Analyzer
├── advanced_plotting.py            # Erweiterte Visualisierungen
├── experiment_configs.yaml         # Konfiguration der Experimente
├── experiment_collection.py        # Datensammlung (bestehend)
└── README_ANALYSIS.md              # Diese Dokumentation
```

## 🚀 **Schnellstart**

### 1. Einfache Ausführung
```bash
cd src/analyse
python config_based_analyzer.py
```

### 2. Programmatische Verwendung
```python
from analyse.config_based_analyzer import ConfigBasedAnalyzer

analyzer = ConfigBasedAnalyzer()

# Nur ResNet + XAI-Methoden
analyzer.analyze_resnet_xai_methods()

# Nur Modelle + GradCAM  
analyzer.analyze_models_gradcam()

# Beide Analysen
analyzer.run_all_analyses()
```

## 📊 **Verfügbare Analysen**

### **1. ResNet + XAI-Methoden (`resnet_xai_methods`)**
Vergleicht verschiedene XAI-Methoden auf dem gleichen Modell:
- GradCAM, Guided Backprop, Integrated Gradients, ScoreCAM
- Fokus auf Explainer-Performance
- Output: `results/resnet_xai_comparison/`

### **2. Modelle + GradCAM (`models_gradcam`)**
Vergleicht verschiedene Modelle mit der gleichen XAI-Methode:
- ResNet18, ResNet50, ResNet101, VGG16 + GradCAM
- Fokus auf Modell-Performance
- Output: `results/models_gradcam_comparison/`

## 🎨 **Verfügbare Plot-Typen**

### **Überblick-Plots**
- **`dataset_overview`**: 6-teiliges Dashboard mit Datenübersicht
- **`accuracy_heatmap`**: Accuracy als farbkodierte Heatmap
- **`sample_distribution`**: Verteilung der Samples pro Modell/Explainer

### **Metrik-Vergleiche**
- **`radar_chart`**: Multi-dimensionaler Metrik-Vergleich
- **`metric_rankings`**: Ranking-Analyse der Metriken
- **`pairwise_comparison`**: Paarweise Metrik-Vergleiche
- **`metric_stability`**: Stabilität und Varianz der Metriken

### **Korrelations-Analysen**
- **`correlation_heatmap`**: Korrelationsmatrix als Heatmap
- **`correlation_network`**: Netzwerk-Darstellung von Korrelationen
- **`metric_vs_accuracy`**: Scatter-Plots Metriken vs. Accuracy

### **Performance-Analysen**
- **`performance_dashboard`**: 4-teiliges Performance-Dashboard
- **`confidence_analysis`**: Analyse der Vorhersage-Confidence
- **`error_analysis`**: Detaillierte Fehler-Analyse

### **Verteilungs-Analysen**
- **`metric_distributions`**: Violin + Box Plots für alle Metriken
- **`statistical_comparison`**: Statistische Tests und Vergleiche
- **`outlier_analysis`**: Erkennung und Analyse von Ausreißern

### **Erweiterte Analysen**
- **`pca_analysis`**: Principal Component Analysis
- **`clustering_analysis`**: Clustering der Experiment-Ergebnisse
- **`model_ranking`**: Ranking-System für Modelle/Explainer

## ⚙️ **Konfiguration**

### **Experiment-Listen anpassen**

Bearbeite `experiment_configs.yaml`:

```yaml
resnet_xai_methods:
  experiments:
    - config_resnet50_grad_cam
    - config_resnet50_guided_backprop
    - config_resnet50_integrated_gradients
    - config_resnet50_score_cam
    # Neue Experimente hinzufügen:
    # - config_resnet50_lime
    
models_gradcam:
  experiments:
    - config_resnet18_grad_cam
    - config_resnet50_grad_cam
    - config_resnet101_grad_cam
    - config_vgg16_grad_cam
    # Neue Modelle hinzufügen:
    # - config_efficientnet_grad_cam
```

### **Plot-Einstellungen**

```yaml
plot_settings:
  create_advanced_plots: true    # Erweiterte Plots erstellen
  create_classic_plots: true     # Klassische Plots erstellen
  plot_dpi: 300                  # Bildqualität
  figure_format: "png"           # Format (png, svg, pdf)
```

## 📈 **Output-Struktur**

Nach einer Analyse findest du folgende Dateien:

```
results/resnet_xai_comparison/
├── combined_data_resnet_xai_methods.csv    # Alle Rohdaten
├── accuracy_by_model_explainer.csv         # Accuracy-Zusammenfassung
├── correlation_matrix.csv                  # Korrelationsmatrix
├── analysis_summary.txt                    # Text-Zusammenfassung
└── plots/                                  # Alle Visualisierungen
    ├── dataset_overview_dashboard.png      # Daten-Überblick
    ├── accuracy_heatmap.png               # Accuracy-Heatmap
    ├── radar_chart.png                    # Metrik-Radar
    ├── correlation_heatmap.png            # Korrelationen
    ├── performance_dashboard.png          # Performance-Analyse
    ├── metric_distributions.png           # Metrik-Verteilungen
    ├── sample_distribution.png            # Sample-Verteilung
    └── ...                                # Weitere Plots
```

## 🛠️ **Erweiterte Verwendung**

### **Eigene Experimente hinzufügen**

```python
analyzer = ConfigBasedAnalyzer()

# Experiment zu ResNet-XAI hinzufügen
analyzer.add_experiment_to_mode(
    'resnet_xai_methods', 
    'config_resnet50_lime'
)

# Experiment entfernen
analyzer.remove_experiment_from_mode(
    'models_gradcam',
    'config_vgg16_grad_cam'
)
```

### **Verfügbare Experimente anzeigen**

```python
analyzer.print_available_configs()  # Alle verfügbaren Configs
analyzer.print_current_setup()      # Aktuelle Konfiguration
```

### **Nur bestimmte Plots erstellen**

```python
from analyse.advanced_plotting import AdvancedPlotter

# Lade Daten
collection = analyzer.load_experiments_for_mode(AnalysisMode.RESNET_XAI_METHODS)

# Erstelle nur bestimmte Plots
plotter = AdvancedPlotter(collection.df, Path("plots"))
plotter._plot_radar_chart()
plotter._plot_correlation_heatmap()
```

## 🔍 **Beispiel-Analysen**

### **1. Welche XAI-Methode ist am besten?**
```python
analyzer.analyze_resnet_xai_methods()
# Schaue dir an: accuracy_heatmap.png, radar_chart.png
```

### **2. Welches Modell funktioniert am besten mit GradCAM?**
```python
analyzer.analyze_models_gradcam()
# Schaue dir an: performance_dashboard.png, model_ranking.png
```

### **3. Gibt es Korrelationen zwischen Metriken?**
```python
# Beide Analysen ausführen
analyzer.run_all_analyses()
# Schaue dir an: correlation_heatmap.png, metric_vs_accuracy.png
```

## 📊 **Interpretations-Hilfen**

### **Radar-Chart**
- **Größere Fläche** = Bessere Performance
- **Gleichmäßige Form** = Konsistente Performance
- **Spitzen** = Starke Performance in spezifischen Metriken

### **Accuracy-Heatmap**
- **Grün** = Hohe Accuracy
- **Rot** = Niedrige Accuracy
- **Vergleiche Zeilen** = Modell-Performance
- **Vergleiche Spalten** = Explainer-Performance

### **Correlation-Heatmap**
- **Rot** = Positive Korrelation
- **Blau** = Negative Korrelation
- **Weiß** = Keine Korrelation
- **Starke Farben** = Starke Korrelation

## 🚨 **Troubleshooting**

### **Problem: Plots werden nicht erstellt**
```python
# Prüfe verfügbare Metriken
print(collection.df.columns)

# Prüfe Datentypen
print(collection.df.dtypes)
```

### **Problem: Experimente werden nicht gefunden**
```python
# Prüfe verfügbare Configs
analyzer.print_available_configs()

# Prüfe Config-Pfade
print(analyzer.config_dir)
```

### **Problem: Leere Plots**
- Stelle sicher, dass CSV-Dateien Daten enthalten
- Prüfe, ob Metrik-Spalten vorhanden sind
- Verifiziere Datentypen (numerisch vs. string)

## 🎉 **Features**

✅ **20+ verschiedene Plot-Typen**  
✅ **Automatische Datenvalidierung**  
✅ **Flexible Konfiguration**  
✅ **Robuste Fehlerbehandlung**  
✅ **Hochauflösende Outputs**  
✅ **Interaktive CLI**  
✅ **Erweiterte Statistiken**  
✅ **Professionelle Visualisierungen**  

Das neue System bietet dir eine vollständige Analyse-Suite für deine XAI-Experimente mit wissenschaftlich fundierten Visualisierungen und aussagekräftigen Statistiken!
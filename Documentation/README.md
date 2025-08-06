# XAI Evaluation Pipeline - Dokumentation

Diese Dokumentation ist nach funktionalen Kategorien organisiert, um die Architektur und Implementierung der XAI Evaluation Pipeline zu verstehen.

## 📂 Ordnerstruktur

### 🏗️ **architecture/**
Systemarchitektur und Gesamtübersicht
- `componenten_architektur.puml` - Komponentenarchitektur
- `class.puml` - Klassendiagramm
- `general_pipeline.puml` - Allgemeine Pipeline-Architektur
- `overview.puml` - Systemübersicht
- `overview_simplefied.puml` - Vereinfachte Übersicht
- `overview_updated.puml` - Aktualisierte Übersicht
- `detailed_pipeline.puml` - Detaillierte Pipeline
- `deployment.puml` - Deployment-Diagramm

### 🔄 **pipeline/**
Pipeline-Ablauf und Datenfluss
- `pipeline.puml` - Pipeline-Diagramm
- `ablauf_simpliefied.puml` - Vereinfachter Ablauf
- `sequence.puml` - Sequenzdiagramm
- `dataflow_dataclasses.puml` - Datenfluss und Datenklassen

### 📊 **data/**
Datenstrukturen und -verarbeitung
- `data_model_classdiagramm.puml` - Datenmodell-Klassendiagramm

### 📈 **evaluation/**
Evaluationsmetriken und -prozesse
- `activitydiagramm.puml` - Aktivitätsdiagramm
- `evaluation_modul_classdiagramm.puml` - Evaluationsmodul-Klassendiagramm
- `metric_model_classdiagramm.puml` - Metrik-Modell-Klassendiagramm
- `evaluation_pipeline.puml` - Evaluations-Pipeline

### 🤖 **models/**
Modell-Architektur
- `ai_model_classdiagramm.puml` - AI-Modell-Klassendiagramm

### 🔍 **xai/**
XAI-Methoden und -Implementierungen
- `simple_factory_register_pattern.puml` - Factory-Register-Pattern
- `xai_factory_classdiagramm.puml` - XAI-Factory-Klassendiagramm
- `xai_model_classdiagramm.puml` - XAI-Modell-Klassendiagramm
- `png/` - Generierte PNG-Bilder der XAI-Diagramme

### ⚙️ **config/**
Konfiguration und Factory-Pattern
- `model_factory_register/` - Model Factory Registrierung
  - `auto_register.puml`
  - `model_register.puml`
  - `png/` - Generierte Bilder
- `Version2/` - Version 2 Konfigurationsdiagramme
  - `XAI_Pipeline_Architecture.puml`
  - `ymalUml.puml`

### 🎨 **visualization/**
Visualisierungen und generierte Diagramme
- `png/` - Alle generierten PNG-Diagramme der Pipeline

### 📚 **legacy/**
Archivierte und experimentelle Diagramme
- `pipeline vorschläge/` - Pipeline-Vorschläge
  - `hybrid.puml`
  - `stufen.puml`
- `puml/` - Archivierte PUML-Dateien

## 🚀 Verwendung

### PlantUML Diagramme generieren
```bash
# Einzelnes Diagramm
plantuml architecture/general_pipeline.puml

# Alle Diagramme in einem Ordner
plantuml architecture/*.puml

# Mit Ausgabepfad
plantuml -o ../png architecture/*.puml
```

### Empfohlene Reihenfolge zum Verstehen der Architektur
1. **Übersicht**: `architecture/overview_simplefied.puml`
2. **Pipeline-Ablauf**: `pipeline/dataflow_dataclasses.puml`
3. **Datenstrukturen**: `data/data_model_classdiagramm.puml`
4. **XAI-Methoden**: `xai/xai_factory_classdiagramm.puml`
5. **Evaluation**: `evaluation/evaluation_pipeline.puml`

## 📝 Hinweise

- Alle `.puml` Dateien können mit PlantUML gerendert werden
- PNG-Versionen befinden sich in den entsprechenden `png/` Ordnern
- Die Diagramme sind miteinander verknüpft und bauen aufeinander auf
- Bei Änderungen an der Pipeline sollten entsprechende Diagramme aktualisiert werden
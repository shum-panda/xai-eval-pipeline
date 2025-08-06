"""
Script zum Ausführen der erweiterten Analyse mit ydata-profiling Reports.
"""

import logging
from pathlib import Path
from src.analyse.config_based_analyzer import ConfigBasedAnalyzer

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Führt die umfassende Analyse aus."""
    
    # Analyzer erstellen
    analyzer = ConfigBasedAnalyzer()
    
    # 1. ResNet XAI Methods Analysis
    results_resnet = analyzer.analyze_resnet_xai_methods()
    
    # 2. VGG XAI Methods Analysis  
    results_vgg = analyzer.analyze_vgg16_xai_methods()
    
    # 3. Model Comparison with GradCAM
    results_models = analyzer.analyze_models_gradcam()

if __name__ == "__main__":
    main()
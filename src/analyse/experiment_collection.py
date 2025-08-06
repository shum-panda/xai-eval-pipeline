import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

KNOWN_METRICS = [
    "iou",
    "point_game",
    "prediction_confidence",
    "accuracy",
    "pixelprecisionrecall_precision",
    "pixelprecisionrecall_recall",
]


class ExperimentCollection:
    """
    Sammlung von Experiment-Daten (verschiedene Modelle, Erklärer, Datasets)
    für Analyse und Vergleich von XAI-Metriken und Modelloutputs.

    Attributes:
        df (pd.DataFrame): Kombinierter DataFrame aller Experimente.
    """

    def __init__(self, dfs: List[pd.DataFrame]):
        """
        Initialisiert ExperimentCollection mit Liste von DataFrames.

        Args:
            dfs: Liste von DataFrames, jeweils ein Experiment-Datensatz.
        """
        self.df = pd.concat(dfs, ignore_index=True)
        # Boolean und Integer Spalten für richtige Vorhersage
        self.df["prediction_correct"] = (
            self.df["predicted_class"] == self.df["true_label"]
        )
        self.df["prediction_correct_int"] = self.df["prediction_correct"].astype(int)

    @classmethod
    def load_from_csvs(cls, csv_paths: List[Path]) -> "ExperimentCollection":
        """
        Lädt mehrere CSV-Dateien und erstellt eine ExperimentCollection.

        Args:
            csv_paths: Liste von Pfaden zu CSV-Dateien.

        Returns:
            ExperimentCollection mit geladenen Daten.
        """
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            dfs.append(df)
        return cls(dfs)

    def group_accuracy_by_model(self) -> pd.DataFrame:
        """
        Berechnet die mittlere Accuracy gruppiert nach Modell und Erklärer.

        Returns:
            DataFrame mit Spalten ["model_name", "explainer_name", "accuracy"]
        """
        return (
            self.df.groupby(["model_name", "explainer_name"])["prediction_correct_int"]
            .mean()
            .reset_index(name="accuracy")
        )

    def correlation_matrix_all_metrics(self) -> pd.DataFrame:
        """
        Gibt Korrelationsmatrix aller bekannten Metriken zurück.

        Returns:
            DataFrame mit paarweisen Korrelationskoeffizienten.
        """
        metric_cols = [col for col in KNOWN_METRICS if col in self.df.columns]
        corr_df = self.df[metric_cols + ["prediction_correct_int"]].corr()
        return corr_df

    def correlation_between(self, x: str, y: str) -> float:
        """
        Berechnet Pearson-Korrelation zwischen zwei Spalten.

        Args:
            x: Name der ersten Spalte.
            y: Name der zweiten Spalte.

        Returns:
            Korrelationskoeffizient (float).

        Raises:
            ValueError: Falls Spalten nicht vorhanden sind.
        """
        if x not in self.df.columns or y not in self.df.columns:
            raise ValueError(f"{x} or {y} not in DataFrame")
        return self.df[[x, y]].corr().iloc[0, 1]

    def prepare_threshold_bins(self, metric: str, bins: int = 5) -> pd.DataFrame:
        """
        Gruppiert nach Bins eines Metrik-Werts und berechnet Accuracy pro Bin.

        Args:
            metric: Metrik-Spalte zum Binning (z.B. "prediction_confidence").
            bins: Anzahl der Quantil-Bins.

        Returns:
            DataFrame mit ["model_name", "explainer_name", "bin",
            "prediction_correct_int"] gemittelt pro Bin.
        """
        df = self.df.copy()
        df["bin"] = pd.qcut(df[metric], q=bins, duplicates="drop")
        grouped = (
            df.groupby(["model_name", "explainer_name", "bin"])[
                "prediction_correct_int"
            ]
            .mean()
            .reset_index()
        )
        return grouped

    def plot_joint_metric_vs_confidence(
        self, metric: str, hue: Optional[str] = "model_name"
    ) -> plt.Figure:
        """
        Scatterplot eines Metrik-Werts gegen Vorhersage-Confidence.

        Args:
            metric: Metrik-Spalte für y-Achse.
            hue: Spalte für Farbgebung (default: "model_name").

        Returns:
            matplotlib Figure Objekt.
        """
        if (
            metric not in self.df.columns
            or "prediction_confidence" not in self.df.columns
        ):
            raise ValueError("Metric or prediction_confidence not in DataFrame")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=self.df, x="prediction_confidence", y=metric, hue=hue, ax=ax
        )
        ax.set_title(f"{metric} vs. Prediction Confidence")
        return fig

    def plot_accuracy_by_model(self) -> plt.Figure:
        """
        Barplot der Accuracy pro Modell und Erklärer.

        Returns:
            matplotlib Figure Objekt.
        """
        df_acc = self.group_accuracy_by_model()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=df_acc, x="model_name", y="accuracy", hue="explainer_name", ax=ax
        )
        ax.set_title("Accuracy by Model and Explainer")
        ax.set_ylim(0, 1)
        return fig

    def plot_metric_comparison(
        self,
        metric: str,
        save_dir: Path,
        figsize=(12, 8),  # Größer für bessere Lesbarkeit
        dpi=150,
        hue: str = "explainer_name",
        order_models: list[str] | None = None,
        order_explainers: list[str] | None = None,
    ) -> Path:
        """
        Erzeugt einen Federboxplot (Violin + Boxplot Overlay) zum Vergleich
        einer Metrik über Modelle und Erklärer mit verbesserter Lesbarkeit.

        Args:
            metric: Name der Metrik-Spalte (z.B. 'iou').
            save_dir: Pfad, um den Plot als PNG zu speichern.
            figsize: Plot-Größe (vergrößert für bessere Lesbarkeit).
            dpi: Bildqualität.
            hue: Gruppierung für Farbgebung (default: explainer_name).
            order_models: Optional Liste der Modellnamen in gewünschter Reihenfolge.
            order_explainers: Optional Liste der Erklärernamen für Farb-Reihenfolge.

        Returns:
            Pfad der gespeicherten PNG-Datei.
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame.")

        df = self.df.copy()

        # Erkennung ob Methodenvergleich oder Modellvergleich
        unique_models = df["model_name"].nunique()
        unique_explainers = df["explainer_name"].nunique()

        # Wenn mehr Explainer als Modelle -> Methodenvergleich
        is_method_comparison = unique_explainers > unique_models

        if is_method_comparison:
            # Für Methodenvergleich: x-Achse = explainer_name, hue = model_name (falls mehrere)
            x_var = "explainer_name"
            hue_var = "model_name" if unique_models > 1 else None
        else:
            # Für Modellvergleich: x-Achse = model_name, hue = explainer_name
            x_var = "model_name"
            hue_var = hue

        # Sortierung setzen falls angegeben
        if order_models is not None:
            df["model_name"] = pd.Categorical(
                df["model_name"], categories=order_models, ordered=True
            )
        if order_explainers is not None:
            df["explainer_name"] = pd.Categorical(
                df["explainer_name"], categories=order_explainers, ordered=True
            )

        # Stil-Konfiguration für bessere Lesbarkeit
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=figsize)

        # Schriftgrößen erhöhen
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
            }
        )

        # Erst Violin Plot als Hintergrund für Verteilungsform
        if len(df[metric].dropna()) > 10:  # Nur wenn genug Daten vorhanden
            violin_plot = sns.violinplot(
                data=df,
                x=x_var,
                y=metric,
                hue=hue_var,
                split=False,  # Nicht splitten
                inner=None,  # Keine inneren Markierungen
                cut=0,
                scale="area",  # Gleiche Fläche für bessere Vergleichbarkeit
                palette="Set2",
                alpha=0.3,  # Sehr transparent als Hintergrund
                linewidth=0,  # Keine Umrandung
                ax=ax,
            )

        # Dann Boxplot mit exakt gleichen Parametern für perfekte Überlagerung
        box_plot = sns.boxplot(
            data=df,
            x=x_var,
            y=metric,
            hue=hue_var,
            showcaps=True,
            boxprops={"facecolor": "white", "linewidth": 1.2, "alpha": 0.9},
            showfliers=True,  # Outliers anzeigen
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            medianprops={"linewidth": 2.5, "color": "red"},
            flierprops={"marker": "o", "alpha": 0.6, "markersize": 4},
            width=0.6,  # Breiter für bessere Sichtbarkeit
            dodge=True,
            ax=ax,
        )

        # Titel und Labels basierend auf Vergleichstyp
        if is_method_comparison:
            title = f"Methodenvergleich: {metric}"
            xlabel = "XAI-Methode"
        else:
            title = f"Modellvergleich: {metric}"
            xlabel = "Modell"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")

        # Display-Namen für Plots beibehalten
        display_name = metric
        if metric == "pixelprecisionrecall_precision":
            display_name = "pixel precision"
        elif metric == "pixelprecisionrecall_recall":
            display_name = "pixel recall"
        else:
            display_name = metric.replace("_", " ").title()

        ax.set_ylabel(display_name, fontsize=14, fontweight="bold")

        # X-Achsen-Labels rotieren für bessere Lesbarkeit
        ax.tick_params(axis="x", rotation=45, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

        # X-Achsen-Labels kürzen falls zu lang
        labels = ax.get_xticklabels()
        shortened_labels = []
        for label in labels:
            text = label.get_text()
            # Kürze lange Namen (sowohl Modell- als auch Methodennamen)
            if len(text) > 15:
                if is_method_comparison:
                    # Kürze Methodennamen
                    text = (
                        text.replace("guided_backprop", "GuidedBP")
                        .replace("integrated_gradients", "IntGrad")
                        .replace("grad_cam", "GradCAM")
                        .replace("score_cam", "ScoreCAM")
                    )
                else:
                    # Kürze Modellnamen
                    text = (
                        text.replace("resnet", "RN")
                        .replace("_grad", "_g")
                        .replace("_integrated", "_int")
                    )
            shortened_labels.append(text)
        ax.set_xticklabels(shortened_labels)

        # Legende verbessern
        handles, labels = ax.get_legend_handles_labels()
        # Entferne doppelte Legende-Einträge (durch split violin plot)
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        # Legende nur anzeigen wenn hue_var gesetzt ist (mehrere Gruppen)
        if hue_var is not None:
            ax.legend(
                unique_handles,
                unique_labels,
                title=hue_var.replace("_", " ").title(),
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=11,
                title_fontsize=12,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        # Grid für bessere Lesbarkeit
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        # Layout anpassen
        plt.tight_layout()

        # Zusätzlichen Platz für rotierte Labels
        plt.subplots_adjust(bottom=0.15, right=0.85)

        save_path = save_dir / f"{metric}_vergleich_federboxplot.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

        # Reset rcParams
        plt.rcParams.update(plt.rcParamsDefault)

        logger = logging.getLogger(__name__)
        logger.info(f"Plot saved: {save_path.resolve()}")
        return save_path

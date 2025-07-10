import os
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
import torch

from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult


class ResultManager:
    def __init__(self, attribution_dir="results/attributions"):
        self.results_per_step: Dict[str, List[XAIExplanationResult]] = {}
        self.dataframes_per_step: Dict[str, pd.DataFrame] = {}
        self.attribution_dir = attribution_dir
        os.makedirs(self.attribution_dir, exist_ok=True)

    def add_results(self, step_name: str, new_results: List[XAIExplanationResult]):
        if step_name not in self.results_per_step:
            self.results_per_step[step_name] = []
        self.results_per_step[step_name].extend(new_results)

    def build_dataframe_for_step(self, step_name: str) -> pd.DataFrame:
        if (
            step_name not in self.results_per_step
            or not self.results_per_step[step_name]
        ):
            raise ValueError(f"No results found for step '{step_name}'.")
        records = []
        for r in self.results_per_step[step_name]:
            attribution_dir = os.path.join(
                self.attribution_dir, r.model_name, r.explainer_name
            )
            os.makedirs(attribution_dir, exist_ok=True)

            attribution_filename = f"{r.image_name}_attribution.pt"
            attribution_path = os.path.join(attribution_dir, attribution_filename)

            torch.save(r.attribution, attribution_path)

            # hole Dict ohne Tensoren
            data = r.to_dict()
            data["attribution_path"] = attribution_path
            del data["attribution"]

            # Optional: direkt hier loggen
            mlflow.log_artifact(
                attribution_path,
                artifact_path=f"attributions/{r.model_name}/{r.explainer_name}",
            )

            records.append(data)
        df = pd.DataFrame(records)
        self.dataframes_per_step[step_name] = df
        return df

    def get_dataframe(self, step_name: str) -> pd.DataFrame:
        """
        Gibt den DataFrame eines Schrittes zurück, baut ihn bei Bedarf.
        """
        if step_name not in self.dataframes_per_step:
            return self.build_dataframe_for_step(step_name)
        return self.dataframes_per_step[step_name]

    def save_dataframe(self, step_name: str, path: str):
        """
        Speichert den DataFrame eines Schrittes als CSV.
        """
        df = self.get_dataframe(step_name)
        df.to_csv(path, index=False)

    def reset(self):
        """
        Setzt alle gespeicherten Ergebnisse und DataFrames zurück.
        """
        self.results_per_step.clear()
        self.dataframes_per_step.clear()

    def save_results(self, results: List[XAIExplanationResult], path: Path):
        torch.save(results, path)

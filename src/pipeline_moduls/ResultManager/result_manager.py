import pandas as pd
from typing import List, Optional

from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult


class ResultManager:
    """
    Collects XAIExplanationResults into a central DataFrame.
    Provides methods to access, save, and reset results.
    """

    def __init__(self):
        self.results: List[XAIExplanationResult] = []
        self.dataframe: Optional[pd.DataFrame] = None

    def add_results(self, new_results: List[XAIExplanationResult]):
        """
        Adds a list of XAIExplanationResults to the internal collection.
        """
        self.results.extend(new_results)

    def build_dataframe(self):
        """
        Converts the collected results into a pandas DataFrame.
        """
        if not self.results:
            raise ValueError("No results to build DataFrame from.")
        self.dataframe = pd.DataFrame([r.to_dict() for r in self.results])
        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the current DataFrame, building it first if necessary.
        """
        if self.dataframe is None:
            return self.build_dataframe()
        return self.dataframe

    def save_dataframe(self, path: str):
        """
        Saves the DataFrame as a CSV file.
        """
        if self.dataframe is None:
            self.build_dataframe()
        self.dataframe.to_csv(path, index=False)

    def reset(self):
        """
        Clears all collected results and resets the DataFrame.
        """
        self.results.clear()
        self.dataframe = None

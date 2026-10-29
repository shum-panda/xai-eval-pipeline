import unittest

import pandas as pd

from src.pipe.moduls.single_run_analyse.single_run_analysis import SingleRunAnalyse


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_correct": [True] * 6 + [False] * 6,
            "iou": [0.6, 0.7, 0.65, 0.8, 0.75, 0.7, 0.1, 0.2, 0.15, 0.05, 0.1, 0.2],
            "model_name": "resnet50",
            "explainer_name": "grad_cam",
        }
    )


class TestSplitByCorrectness(unittest.TestCase):
    """Regression: the boolean masks used 'not series', which always raised."""

    def test_statistical_tests_are_computed(self) -> None:
        analysis = SingleRunAnalyse(make_results())
        result = analysis.calculate_statistical_tests_for_all_metrics()
        self.assertFalse(result.empty)

    def test_iou_distribution_stats_are_computed(self) -> None:
        analysis = SingleRunAnalyse(make_results())
        result = analysis.calculate_iou_distribution_stats()
        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main()

import random
import unittest
from pathlib import Path

import numpy as np
import torch

from src.pipe.control.utils.provenance import (
    PROJECT_ROOT,
    collect_provenance,
    resolve_project_path,
    set_seeds,
)


class TestResolveProjectPath(unittest.TestCase):

    def test_project_root_contains_main(self) -> None:
        self.assertTrue((PROJECT_ROOT / "main.py").is_file())

    def test_relative_path_is_resolved_against_project_root(self) -> None:
        self.assertEqual(
            resolve_project_path("data/raw/final_mapping.txt"),
            PROJECT_ROOT / "data" / "raw" / "final_mapping.txt",
        )

    def test_absolute_path_is_kept(self) -> None:
        absolute = Path(PROJECT_ROOT.anchor) / "somewhere" / "images"
        self.assertEqual(resolve_project_path(str(absolute)), absolute)


class TestSetSeeds(unittest.TestCase):

    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)

    def _draw(self) -> tuple:
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    def test_same_seed_gives_same_numbers(self) -> None:
        set_seeds(123, deterministic=False)
        first = self._draw()
        set_seeds(123, deterministic=False)
        self.assertEqual(first, self._draw())

    def test_different_seed_gives_different_numbers(self) -> None:
        set_seeds(1, deterministic=False)
        first = self._draw()
        set_seeds(2, deterministic=False)
        self.assertNotEqual(first, self._draw())

    def test_deterministic_mode_is_enabled(self) -> None:
        set_seeds(0, deterministic=True)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())


class TestCollectProvenance(unittest.TestCase):

    def test_contains_code_and_environment_information(self) -> None:
        provenance = collect_provenance()
        for key in (
            "git_commit",
            "git_branch",
            "git_dirty",
            "python_version",
            "torch_version",
            "torchvision_version",
            "platform",
        ):
            self.assertIn(key, provenance)
        self.assertEqual(provenance["torch_version"], torch.__version__)

    def test_git_commit_is_a_full_sha_inside_the_repository(self) -> None:
        commit = collect_provenance()["git_commit"]
        if commit == "unknown":
            self.skipTest("git not available")
        self.assertRegex(commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()

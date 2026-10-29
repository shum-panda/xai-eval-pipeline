import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient
from test_ui_workflow import graph

from src.ui.runner import JobManager
from src.ui.server import create_app


def fake_command(config_dir: Path, config_name: str) -> List[str]:
    """Stands in for main.py: prints which config it got and succeeds."""
    script = (
        "import sys; print('running', sys.argv[1]);"
        "sys.exit(1 if 'vgg16' in sys.argv[1] else 0)"
    )
    return [sys.executable, "-c", script, str(config_dir / f"{config_name}.yaml")]


class TestServer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        jobs = JobManager(command=fake_command, generated_dir=self.temp_dir / "gen")
        app = create_app(
            job_manager=jobs,
            workflows_dir=self.temp_dir / "workflows",
            frontend_dist=self.temp_dir / "no_frontend",
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_catalog(self) -> None:
        response = self.client.get("/api/catalog")
        self.assertEqual(response.status_code, 200)
        self.assertIn("xai", response.json()["kinds"])

    def test_compile_returns_yaml_per_run(self) -> None:
        response = self.client.post(
            "/api/compile", json=graph(["resnet18", "resnet34"], ["grad_cam"])
        )
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["runs"]), 2)
        self.assertIn("name: resnet18", body["runs"][0]["yaml"])

    def test_invalid_workflow_is_not_started(self) -> None:
        response = self.client.post(
            "/api/runs", json={"name": "x", "nodes": [], "edges": []}
        )
        self.assertEqual(response.status_code, 400)

    def test_runs_execute_one_after_another_and_report_status(self) -> None:
        response = self.client.post(
            "/api/runs", json=graph(["resnet18", "vgg16"], ["grad_cam"])
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["id"]

        for _ in range(100):
            job = self.client.get(f"/api/runs/{job_id}").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)

        self.assertEqual(job["status"], "finished")
        statuses = {run["name"]: run["status"] for run in job["runs"]}
        self.assertEqual(statuses["test_flow_resnet18_grad_cam"], "succeeded")
        self.assertEqual(statuses["test_flow_vgg16_grad_cam"], "failed")
        self.assertTrue(any(line.startswith("running") for line in job["log"]))
        written = sorted(
            p.name for p in (self.temp_dir / "gen" / "test_flow").iterdir()
        )
        self.assertEqual(
            written,
            ["test_flow_resnet18_grad_cam.yaml", "test_flow_vgg16_grad_cam.yaml"],
        )

    def test_workflow_save_and_load(self) -> None:
        flow = graph(["resnet50"], ["grad_cam"])
        self.client.put("/api/workflows/My Flow", json=flow)

        self.assertEqual(self.client.get("/api/workflows").json(), ["My_Flow"])
        loaded = self.client.get("/api/workflows/My_Flow").json()
        self.assertEqual(loaded["nodes"], flow["nodes"])

    def test_root_explains_how_to_build_frontend(self) -> None:
        response = self.client.get("/")
        self.assertIn("npm run build", response.text)


if __name__ == "__main__":
    unittest.main()

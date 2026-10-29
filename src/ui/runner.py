"""
Execute compiled workflows in the background.

Each run is started exactly like a manual run, as `python main.py
--config-dir <dir> --config-name <run>` with the generated YAML. The UI adds no
execution path of its own, so every result can be repeated without the UI.
"""

import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from src.pipe.control.utils.provenance import PROJECT_ROOT
from src.ui.workflow import RunConfig

GENERATED_DIR = PROJECT_ROOT / "config" / "generated"
MAX_LOG_LINES = 2000

#: Builds the command for one run; replaceable in tests.
CommandFactory = Callable[[Path, str], List[str]]


def default_command(config_dir: Path, config_name: str) -> List[str]:
    return [
        sys.executable,
        "main.py",
        "--config-dir",
        str(config_dir),
        "--config-name",
        config_name,
    ]


@dataclass
class RunState:
    name: str
    model: str
    xai: str
    config_path: str
    status: str = "pending"  # pending | running | succeeded | failed | cancelled
    return_code: Optional[int] = None


@dataclass
class Job:
    id: str
    workflow: str
    runs: List[RunState]
    status: str = "pending"  # pending | running | finished | cancelled
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    cancel_requested: bool = False
    process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]

    def to_dict(self, log_lines: int = 300) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow": self.workflow,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runs": [vars(run) for run in self.runs],
            "log": list(self.log)[-log_lines:],
        }


def write_configs(
    workflow_slug: str, runs: List[RunConfig], generated_dir: Path = GENERATED_DIR
) -> Path:
    """Write one YAML per run to <generated_dir>/<workflow>/ and return the dir."""
    config_dir = generated_dir / workflow_slug
    config_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (config_dir / f"{run.name}.yaml").write_text(run.yaml, encoding="utf-8")
    return config_dir


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class JobManager:
    """Runs one job at a time; a job executes its runs one after another."""

    def __init__(
        self,
        command: CommandFactory = default_command,
        generated_dir: Path = GENERATED_DIR,
    ) -> None:
        self._command = command
        self._generated_dir = generated_dir
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, workflow_slug: str, runs: List[RunConfig]) -> Job:
        with self._lock:
            if any(job.status == "running" for job in self._jobs.values()):
                raise RuntimeError("A workflow is already running.")
            config_dir = write_configs(workflow_slug, runs, self._generated_dir)
            job = Job(
                id=uuid.uuid4().hex[:12],
                workflow=workflow_slug,
                runs=[
                    RunState(
                        name=run.name,
                        model=run.model,
                        xai=run.xai,
                        config_path=_display_path(config_dir / f"{run.name}.yaml"),
                    )
                    for run in runs
                ],
                status="running",
            )
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._execute, args=(job, config_dir), daemon=True
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.cancel_requested = True
        if job.process is not None and job.process.poll() is None:
            job.process.terminate()
        return job

    def _execute(self, job: Job, config_dir: Path) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        for run in job.runs:
            if job.cancel_requested:
                run.status = "cancelled"
                continue
            run.status = "running"
            job.log.append(f"=== {run.name} ({run.config_path}) ===")
            try:
                job.process = subprocess.Popen(
                    self._command(config_dir, run.name),
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert job.process.stdout is not None
                for line in job.process.stdout:
                    job.log.append(line.rstrip("\n"))
                run.return_code = job.process.wait()
            except OSError as error:
                job.log.append(f"Could not start run: {error}")
                run.return_code = -1
            if job.cancel_requested:
                run.status = "cancelled"
            else:
                run.status = "succeeded" if run.return_code == 0 else "failed"
            job.log.append(f"=== {run.name}: {run.status} ===")

        job.process = None
        job.status = "cancelled" if job.cancel_requested else "finished"
        job.finished_at = datetime.now().isoformat()

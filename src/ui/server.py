"""
HTTP API and static file server for the pipeline UI.

Start with:  python -m src.ui.server   (then open http://127.0.0.1:8000)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.pipe.control.utils.config_dataclasses.config_schema import (
    register_config_schema,
)
from src.pipe.control.utils.provenance import PROJECT_ROOT
from src.ui.catalog import build_catalog
from src.ui.runner import JobManager
from src.ui.workflow import CompileResult, Workflow, compile_workflow, slug

WORKFLOWS_DIR = PROJECT_ROOT / "config" / "workflows"
FRONTEND_DIST = PROJECT_ROOT / "ui" / "dist"

NOT_BUILT_PAGE = """<!doctype html><meta charset="utf-8"><title>Pipeline UI</title>
<body style="font-family:sans-serif;max-width:40rem;margin:4rem auto">
<h1>Frontend not built</h1>
<p>The API is running. Build the frontend once with</p>
<pre>cd ui
npm install
npm run build</pre>
<p>or start the dev server with <code>npm run dev</code> in <code>ui/</code>.</p>
</body>"""


def _compile_payload(result: CompileResult) -> Dict[str, Any]:
    return {
        "ok": result.ok,
        "errors": result.errors,
        "runs": [
            {"name": r.name, "model": r.model, "xai": r.xai, "yaml": r.yaml}
            for r in result.runs
        ],
    }


def create_app(
    job_manager: Optional[JobManager] = None,
    workflows_dir: Path = WORKFLOWS_DIR,
    frontend_dist: Path = FRONTEND_DIST,
) -> FastAPI:
    """Build the app; dependencies are injectable for tests."""
    register_config_schema()
    jobs = job_manager or JobManager()
    catalog = build_catalog()
    app = FastAPI(title="XAI Pipeline UI")

    @app.get("/api/catalog")
    def get_catalog() -> Dict[str, Any]:
        return catalog

    @app.post("/api/compile")
    def compile_graph(workflow: Dict[str, Any]) -> Dict[str, Any]:
        return _compile_payload(compile_workflow(Workflow.from_dict(workflow)))

    @app.post("/api/runs")
    def start_runs(workflow: Dict[str, Any]) -> Dict[str, Any]:
        parsed = Workflow.from_dict(workflow)
        result = compile_workflow(parsed)
        if not result.ok:
            raise HTTPException(status_code=400, detail=_compile_payload(result))
        try:
            job = jobs.start(slug(parsed.name), result.runs)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return job.to_dict()

    @app.get("/api/runs")
    def list_runs() -> List[Dict[str, Any]]:
        return [job.to_dict(log_lines=0) for job in jobs.list()]

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> Dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job.to_dict()

    @app.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str) -> Dict[str, Any]:
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job.to_dict()

    @app.get("/api/workflows")
    def list_workflows() -> List[str]:
        if not workflows_dir.is_dir():
            return []
        return sorted(p.stem for p in workflows_dir.glob("*.json"))

    @app.get("/api/workflows/{name}")
    def load_workflow(name: str) -> Dict[str, Any]:
        path = workflows_dir / f"{slug(name)}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown workflow")
        workflow: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return workflow

    @app.put("/api/workflows/{name}")
    def save_workflow(name: str, workflow: Dict[str, Any]) -> Dict[str, str]:
        workflows_dir.mkdir(parents=True, exist_ok=True)
        workflow = {**workflow, "name": slug(name)}
        path = workflows_dir / f"{slug(name)}.json"
        path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
        return {"name": slug(name)}

    if (frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")
    else:

        @app.get("/", response_class=HTMLResponse)
        def frontend_missing() -> str:
            return NOT_BUILT_PAGE

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the XAI pipeline UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

import mlflow
import hydra

from src.pipeline.control.utils import MasterConfig
from src.pipeline.control import XAIOrchestrator


@hydra.main(config_path="configs", config_name="default", version_base=None)
def run(cfg: MasterConfig):
    with mlflow.start_run(run_name=cfg.experiment.name):
        orchestrator = XAIOrchestrator(cfg)
        orchestrator.run()

if __name__ == "__main__":
    run()

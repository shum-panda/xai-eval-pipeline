import mlflow
import hydra

from src.pipe.control.orchestrator import Orchestrator
from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig




@hydra.main(config_path="configs", config_name="default", version_base=None)
def run(cfg: MasterConfig):
    with mlflow.start_run(run_name=cfg.experiment.name):
        orchestrator = Orchestrator(cfg)
        orchestrator.run()

if __name__ == "__main__":
    run()

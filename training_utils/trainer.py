import os
from datetime import datetime

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def get_default_root_dir(stage_dir):
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(stage_dir, "root_dir", os.environ["SLURM_JOB_ID"])
    else:
        return os.path.join(
            stage_dir, "root_dir", datetime.now().strftime("%Y-%m-%d--%H-%M")
        )

class Trainer(L.Trainer):

    def __init__(self, stage_dir: str="", run_name: str=None, **kwargs):

        stage_dir = os.path.join(stage_dir, run_name) if isinstance(run_name, str) else stage_dir
        default_root_dir = get_default_root_dir(stage_dir)

        if kwargs.get("fast_dev_run"):
            default_root_dir = "/tmp"
        else:
            os.makedirs(default_root_dir, exist_ok=True)

        rank_zero_info(f"Setting default root dir: {default_root_dir}")

        super().__init__(
            default_root_dir=default_root_dir,
            **kwargs
        )
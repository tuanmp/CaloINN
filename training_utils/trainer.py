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
        return os.path.join(stage_dir, os.environ["SLURM_JOB_ID"])
    else:
        return os.path.join(
            stage_dir, datetime.now().strftime("%Y-%m-%d--%H-%M")
        )

class Trainer(L.Trainer):

    def __init__(self, stage_dir: str="", run_name: str=None, from_slurm_id: int=None, **kwargs):

        # the from_slurm_id argument should only be used to make predictions from a previous slurm job
        # if from_slurm_id is not None, we will use the slurm job id as the run name to load the checkpoint from that job

        stage_dir = os.path.join(stage_dir, run_name) if isinstance(run_name, str) else stage_dir
        default_root_dir = get_default_root_dir(stage_dir)

        if kwargs.get("fast_dev_run"):
            default_root_dir = "/tmp"
        else:
            os.makedirs(default_root_dir, exist_ok=True)

        if from_slurm_id is not None:
            default_root_dir = os.path.join(stage_dir, str(from_slurm_id))
            if not os.path.exists(default_root_dir):
                raise ValueError(f"Checkpoint directory for slurm job id {from_slurm_id} does not exist at {default_root_dir}")
            else:
                rank_zero_info(f"Loading checkpoint from slurm job id {from_slurm_id} at {default_root_dir}")
        else:
            rank_zero_info(f"Setting default root dir: {default_root_dir}")

        super().__init__(
            default_root_dir=default_root_dir,
            **kwargs
        )

        self.run_name = run_name
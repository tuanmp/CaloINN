import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from lightning_fabric.utilities import rank_zero_info

from src.caloch_eval import evaluate

plt.style.use(["science"])

figsize = (8, 6)


class ResultPlotter(L.Callback):

    def __init__(self, reference_file, prediction_file):
        super().__init__()
        self.plot_dir = "./plots"
        self.reference_file = reference_file
        self.prediction_file = prediction_file
    
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        
        super().setup(trainer, pl_module, stage)

        self.plot_dir = os.path.join(trainer.default_root_dir, "results")

        if trainer.is_global_zero:
            os.makedirs(self.plot_dir, exist_ok=True)

    def on_predict_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        
        rank_zero_info(f"Generating result plots to {self.plot_dir}...")
        rank_zero_info("Using the following command to evaluate results:")
        rank_zero_info(
             f"evaluate -i {os.path.join(trainer.default_root_dir, 'predictions', self.prediction_file)} "
             f"-r {self.reference_file} "
             f"-m all -d {trainer.datamodule.eval_dataset} "
             f"--output_dir {self.plot_dir}"
        )
        evaluate.main(
            (
                f"-i {os.path.join(trainer.default_root_dir, 'predictions', self.prediction_file)} "
                f"-r {self.reference_file} "
                f"-m all -d {trainer.datamodule.eval_dataset} "
                f"--output_dir {self.plot_dir}"
            ).split()
        )

        rank_zero_info(f"Finished generating result plots to {self.plot_dir}.")
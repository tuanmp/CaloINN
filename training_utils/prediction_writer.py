import os
import shutil

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class BaseWriter(L.pytorch.callbacks.BasePredictionWriter):

    def __init__(self, predict_output_file="samples.hdf5"):

        super().__init__()

        self.save_dir = "./predictions"
        self.predict_output_file = predict_output_file
        self.batch_dir = os.path.join(self.save_dir, "batches")

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):

        super().setup(trainer, pl_module, stage)

        if stage != "predict":
            return

        # if self.save_dir is None:
        #     self.save_dir = os.path.join(trainer.default_root_dir, "predictions")
        #     self.batch_dir = os.path.join(self.save_dir, "batches")

        if trainer.is_global_zero:
            self.save_dir = os.path.join(trainer.default_root_dir, "predictions")
            self.batch_dir = os.path.join(self.save_dir, "batches")
            os.makedirs(self.save_dir, exist_ok=True)
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

            # make a subdirectory to store batches, then combine them at the end of prediction
            os.makedirs(self.batch_dir, exist_ok=True)

        rank_zero_info(f"PredictionWriter will save predictions to {self.save_dir}")

    def save_batch(self, incident_energies, showers, batch_idx: str, dataloader_idx: str):

        save_path = os.path.join(self.batch_dir, f"dl_{dataloader_idx}_batch_{batch_idx}.npz")

        np.savez(save_path, incident_energies=incident_energies, showers=showers)


class PredictionWriter(BaseWriter):

    def __init__(self, predict_output_file="samples.hdf5"):
        super().__init__(predict_output_file)

        self.inference_time = []

    def write_on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        incident_energies, showers, t = prediction

        self.save_batch(
            incident_energies=incident_energies,
            showers=showers,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        self.inference_time.append(t)

    def on_predict_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):

        # combine batches into one hdf5 file and save to self.save_dir
        rank_zero_info("Combining batch predictions into one file...")
        if trainer.is_global_zero:
            # combine batches into one file

            batch_dir = self.batch_dir
            batch_files = sorted(os.listdir(batch_dir))

            num_predictions = 0
            with h5py.File(os.path.join(self.save_dir, self.predict_output_file), "w") as f:

                for batch_file in batch_files:
                    batch_path = os.path.join(batch_dir, batch_file)
                    data = np.load(batch_path, allow_pickle=True)

                    incident_energies = data["incident_energies"]
                    showers = data["showers"]

                    if num_predictions == 0:
                        f.create_dataset("incident_energies", data=incident_energies, maxshape=(None,1), chunks=True)
                        f.create_dataset("showers", data=showers, maxshape=(None, *showers.shape[1:]), chunks=True)
                    else:
                        f["incident_energies"].resize(num_predictions + incident_energies.shape[0], axis=0)
                        f["incident_energies"][-incident_energies.shape[0]:] = incident_energies

                        f["showers"].resize(num_predictions + showers.shape[0], axis=0)
                        f["showers"][-showers.shape[0]:] = showers

                    num_predictions += incident_energies.shape[0]
        
        rank_zero_info(f"Saved {num_predictions} showers to {os.path.join(self.save_dir, self.predict_output_file)}")
        rank_zero_info(f"Average inference time per sample: {np.mean(self.inference_time):.6f} seconds")
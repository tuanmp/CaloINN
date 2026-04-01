import math
import time
from random import randrange
from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.distributions as dist
from lightning_fabric.utilities import rank_zero_info

import caloch_eval.evaluate as evaluate
import data_util
from model import CINN
from training_diagnostics import TrainingDiagnostics


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(
            dist.Uniform(torch.log(lb), torch.log(ub)),
            dist.ExpTransform(),
        )


class CaloINNLightningModule(pl.LightningModule):
    """Lightning module that preserves the CINN training logic."""

    def __init__(
        self,
        setup_data_sample_path,
        enable_diagnostics=False,
        actnorm_calibration_samples=1024,
        xml_path="./binning_dataset_1_pions.xml",
        xml_ptype="pion",
        dataset_params={},
        cinn_params={},
        width_noise=1e-7,
        custom_noise=False,
        single_energy=None,
    ):
        super().__init__()

        self.cinn_params = cinn_params
        self.width_noise = width_noise
        self.batch_number = 0
        self._run_batch_diagnostics = False
        self._diag_batch_idx = -1
        self.diagnostics = TrainingDiagnostics(enabled=enable_diagnostics)

        self.save_hyperparameters()

        assert setup_data_sample_path is not None, "Must provide setup_data_sample_path to initialize model parameters"
        assert setup_data_sample_path.endswith(".hdf5"), "setup_data_sample_path must be an .hdf5 file path"

        sample_x, sample_c, self.layer_boundaries = self.load_init_tensors()
        self.num_dim = int(sample_x.shape[1])

        n_calib = min(int(actnorm_calibration_samples), int(sample_x.shape[0]))
        self._actnorm_calib_x = sample_x[:n_calib].detach().clone()
        self._actnorm_calib_c = sample_c[:n_calib].detach().clone()

        self.num_train_samples = 1

        if self.hparams["custom_noise"]:
            q = self.eval_quantiles(torch.clone(sample_x))
        else:
            q = torch.tensor(self.width_noise, dtype=sample_x.dtype)

        self.register_buffer("q", q)

        self.model = CINN(self.cinn_params, sample_x, sample_c)

    def load_init_tensors(self):

        rank_zero_info(f"Loading sample data from {self.hparams['setup_data_sample_path']} to initialize model parameters...")
        
        sample_data, layer_boundaries = data_util.load_data(
            self.hparams["setup_data_sample_path"],
            self.hparams["xml_ptype"],
            self.hparams["xml_path"]
        )

        x, c = data_util.preprocess(
            sample_data,
            layer_boundaries,
            self.hparams["dataset_params"].get("eps", 1.0e-10),
            u0up_cut=self.hparams["dataset_params"].get("u0up_cut", 7.0),
            u0low_cut=self.hparams["dataset_params"].get("u0low_cut", 0.0),
            rew=self.hparams["dataset_params"].get("pt_rew", 1.0),
            dep_cut=self.hparams["dataset_params"].get("dep_cut", 1.0e10),
        )

        dtype = torch.get_default_dtype()
        x = torch.tensor(x, dtype=dtype)
        c = torch.tensor(c, dtype=dtype)
        return x, c, layer_boundaries

    def setup(self, stage: str):

        if stage == "fit":
            self.num_train_samples = self.trainer.datamodule.num_train_samples
            rank_zero_info(f"Number of training samples: {self.num_train_samples}")

    def eval_quantiles(self, data):
        cp = torch.clone(data)
        cp[cp == 0] = torch.nan
        quantiles = torch.nanquantile(cp, q=0.01, dim=0).reshape(1, -1)
        return quantiles

    def get_actnorm_calibration_batch(self, max_samples=None):
        if not hasattr(self, "_actnorm_calib_x") or not hasattr(self, "_actnorm_calib_c"):
            return None
        n = self._actnorm_calib_x.shape[0]
        if max_samples is not None:
            n = min(int(max_samples), int(n))
        return self._actnorm_calib_x[:n], self._actnorm_calib_c[:n]


    def _compute_losses(self, x, c, run_diagnostics=False):
        # Diagnostics before log_prob
        log_probs = self.model.log_prob(x, c)
        if run_diagnostics:
            self.diagnostics.log_loss_diagnostics(self.model, x, c, log_probs, self.batch_number)
        
        inn_loss = -torch.mean(log_probs)
        if self.model.bayesian:
            kl_loss = self.model.get_kl() / self.num_train_samples
            loss = inn_loss + kl_loss
        else:
            kl_loss = None
            loss = inn_loss
        return loss, inn_loss, kl_loss

    def on_fit_start(self):
        if self.model.bayesian:
            self.model.enable_map()

        rank_zero_info("Starting training...")

    def on_after_backward(self):
        # Restrict gradient diagnostics to the first batch of the first epoch.
        if self._run_batch_diagnostics:
            self.diagnostics.log_gradient_diagnostics(
                self,
                tag=f"after_backward_epoch{self.current_epoch}_batch{self._diag_batch_idx}",
            )

    def on_before_optimizer_step(self, optimizer):
        # Explicit one-time check before the very first optimizer step.
        self.diagnostics.maybe_log_before_first_optimizer_step(self, self._run_batch_diagnostics)

    def training_step(self, batch, batch_idx):
        x, c = batch
        self.batch_number += 1
        self._run_batch_diagnostics = self.diagnostics.should_run_batch(self.current_epoch, batch_idx)
        self._diag_batch_idx = int(batch_idx)
        if self._run_batch_diagnostics:
            self.diagnostics.log_first_block_inside_ratio(self.model, x, c, tag=f"epoch{self.current_epoch}_batch{batch_idx}")
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=self._run_batch_diagnostics)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_idx}: "
                f"loss={loss.detach().cpu().item()}, x_min={x.min().detach().cpu().item()}, "
                f"x_max={x.max().detach().cpu().item()}, "
                f"c_min={c.min().detach().cpu().item()}, c_max={c.max().detach().cpu().item()}"
            )

        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_inn_loss", inn_loss.detach(), on_step=True, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("train_kl_loss", kl_loss.detach(), on_step=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, c = batch
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=False)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss at batch {batch_idx}: "
                f"loss={loss.detach().cpu().item()}, x_min={x.min().detach().cpu().item()}, "
                f"x_max={x.max().detach().cpu().item()}, "
                f"c_min={c.min().detach().cpu().item()}, c_max={c.max().detach().cpu().item()}"
            )

        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_inn_loss", inn_loss.detach(), on_step=False, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("val_kl_loss", kl_loss.detach(), on_step=False, on_epoch=True, batch_size=x.shape[0])

    def test_step(self, batch, batch_idx):
        x, c = batch
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=False)

        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("test_inn_loss", inn_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("test_kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])

    def on_predict_start(self):
        rank_zero_info("Starting prediction...")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Generate once per predict run; Lightning still requires a predict dataloader.
        x, c = batch 

        samples, t = self.conditional_generate(c, measure_gen_time=True)
        self.log("predict_gen_time", t / c.shape[0], on_step=False, on_epoch=True, prog_bar=True, batch_size=c.shape[0])
        samples -= self.width_noise
        shower = data_util.postprocess(
            samples.cpu().numpy(),
            c.cpu().numpy(),
            layer_boundaries=self.layer_boundaries,
            threshold=self.width_noise,
            quantiles=self.q.detach().cpu().numpy(),
        )

        return shower


    @torch.inference_mode()
    def conditional_generate(
        self,
        incident_energies: torch.Tensor,
        measure_gen_time=False,
    ):
        self.model.eval()
        device = self.device

        t1 = time.time()
        samples = self.model.sample(1, incident_energies.to(device))
        t_diff = time.time() - t1

        if measure_gen_time:
            return samples, t_diff
        return samples      

    # def generate(self, num_samples, incident_energies: torch.Tensor=None, batch_size=1000, output_file=None):
    #     self.model.eval()
    #     device = self.device

    #     with torch.no_grad():
    #         if self.hparams["dataset_params"].get("eval_dataset") == "2":
    #             logunif = LogUniform(torch.tensor(1e3), torch.tensor(1e6))
    #             energies = logunif.sample((num_samples, 1)) / 1e3
    #         elif incident_energies is not None:
    #             energies = incident_energies.to(device)
    #         else:
    #             energies = (
    #                 torch.tensor(
    #                     data_util.generate_Einc_ds1(
    #                         energy=self.single_energy,
    #                         sample_multiplier=1000,
    #                     ),
    #                     dtype=torch.float,
    #                 )
    #                 / 1e3
    #             ).reshape(-1, 1)

    #         samples = torch.zeros((energies.shape[0], 1, self.num_dim), device=device)
    #         num_samples = energies.shape[0]
    #         times = []

    #         for batch in range((num_samples + batch_size - 1) // batch_size):
    #             start = batch_size * batch
    #             stop = min(batch_size * (batch + 1), num_samples)
    #             energies_l = energies[start:stop].to(device)
    #             t1 = time.time()
    #             samples[start:stop] = self.model.sample(1, energies_l)
    #             t_diff = time.time() - t1
    #             times.append(t_diff / (stop - start))

    #         self.avg_gen_time[str(batch_size)] = np.array(times).mean()
    #         samples = samples[:, 0, ...].cpu().numpy()
    #         energies = energies.cpu().numpy()

    #     samples -= self.width_noise
    #     shower = data_util.postprocess(
    #         samples,
    #         energies,
    #         layer_boundaries=self.layer_boundaries,
    #         threshold=self.width_noise,
    #         quantiles=self.q.detach().cpu().numpy(),
    #     )

    #     if output_file is not None:
    #         save_payload = {key: np.copy(value) for key, value in shower.items()}
    #         data_util.save_data(save_payload, filename=output_file)

    #     return shower

    # def plot_default_from_caloch(self, base_dir, sample_name="samples.hdf5", eval_name="final", cut=1.515e-3):
    #     evaluate.main(
    #         (
    #             f"-i {base_dir}/{sample_name} "
    #             f"-r {self.params['val_data_path']} "
    #             f"-m all -d {self.params['eval_dataset']} "
    #             f"--output_dir {base_dir}/eval/{eval_name}/ --cut {cut}"
    #         ).split()
    #     )

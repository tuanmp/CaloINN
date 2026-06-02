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


class _SkipLastTwoScheduler:
    """Wrapper that skips the last 2 scheduler.step() calls, matching legacy.

    Trainer.py lines 150-153:
        if i < self.scheduler.total_steps-2:
            self.scheduler.step()
        else:
            pass

    This prevents OneCycleLR from reaching its terminal annealing, freezing
    the LR at the penultimate value for the final 2 optimizer updates.
    """
    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._total = getattr(scheduler, "total_steps", 0)
        self._step_count = 0

    def step(self):
        if self._step_count < self._total - 2:
            self._scheduler.step()
        self._step_count += 1

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, sd):
        self._scheduler.load_state_dict(sd)

    def get_last_lr(self):
        return self._scheduler.get_last_lr()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


class CaloINNLightningModule(pl.LightningModule):
    """Lightning module that preserves the CINN training logic."""

    def __init__(
        self,
        setup_data_sample_path=None,
        enable_diagnostics=False,
        actnorm_calibration_samples=1024,
        xml_path="./binning_dataset_1_pions.xml",
        xml_ptype="pion",
        dataset_params={},
        cinn_params={},
        width_noise=1e-7,
        custom_noise=False,
        single_energy=None,
        train_val_frac=0.01,
        train_batch_size=512,
        train_shuffle=False,
        init_from_legacy_train_split=True,
        optimizer_params=None,
        scheduler_params=None,
        # Phase 2: accept preprocessed data arrays directly, bypassing file read
        init_data_x=None,
        init_data_c=None,
        init_layer_boundaries=None,
        init_num_train_samples=None,
        # max_samples: randomly subsample init data (matches legacy behavior)
        max_samples=None,
    ):
        super().__init__()

        self.cinn_params = cinn_params
        self.optimizer_params = dict(optimizer_params or {})
        self.scheduler_params = dict(scheduler_params or {})
        self.width_noise = width_noise
        self.batch_number = 0
        self._run_batch_diagnostics = False
        self._diag_batch_idx = -1
        self.diagnostics = TrainingDiagnostics(enabled=enable_diagnostics)

        self.save_hyperparameters()

        # Phase 2: Support direct data arrays from DataModule, bypassing
        # redundant HDF5 file reads. Legacy path (via setup_data_sample_path)
        # is kept for backwards compatibility.
        if init_data_x is not None and init_data_c is not None:
            sample_x = init_data_x
            sample_c = init_data_c
            self.layer_boundaries = init_layer_boundaries
        else:
            assert setup_data_sample_path is not None, (
                "Must provide either setup_data_sample_path or init_data_x/init_data_c"
            )
            assert setup_data_sample_path.endswith(".hdf5"), (
                "setup_data_sample_path must be an .hdf5 file path"
            )
            sample_x, sample_c, self.layer_boundaries = self.load_init_tensors()

        self.num_dim = int(sample_x.shape[1])

        # Phase 2b: max_samples — randomly subsample init data to match
        # legacy behavior where only max_samples are used for CINN
        # initialization (trainer.py lines 62-64).  The full dataset
        # count is preserved in num_train_samples for KL scaling.
        n_full_init = int(sample_x.shape[0])
        if max_samples is not None and max_samples > 0 and max_samples < n_full_init:
            torch.manual_seed(42)  # deterministic subsample (legacy uses randperm)
            rand_idx = torch.randperm(n_full_init)[: int(max_samples)]
            sample_x = sample_x[rand_idx]
            sample_c = sample_c[rand_idx]

        n_calib = min(int(actnorm_calibration_samples), int(sample_x.shape[0]))
        self._actnorm_calib_x = sample_x[:n_calib].detach().clone()
        self._actnorm_calib_c = sample_c[:n_calib].detach().clone()

        # Phase 2: num_train_samples is now set from actual data or override.
        # It will be updated by setup() when the datamodule is attached.
        # For bayesian models, the correct value is critical (KL scaling).
        if init_num_train_samples is not None:
            self.num_train_samples = int(init_num_train_samples)
        else:
            # Use full dataset size for KL scaling (n_full_init preserved
            # before max_samples subsampling, matching legacy behavior where
            # N = len(train_loader.data) is the full count).
            self.num_train_samples = max(1, n_full_init)

        if self.hparams["custom_noise"]:
            q = self.eval_quantiles(torch.clone(sample_x))
        else:
            q = torch.tensor(self.width_noise, dtype=sample_x.dtype)

        self.register_buffer("q", q)

        self.model = CINN(self.cinn_params, sample_x, sample_c)

    def load_init_tensors(self):

        rank_zero_info(f"Loading sample data from {self.hparams['setup_data_sample_path']} to initialize model parameters...")
        if self.hparams.get("init_from_legacy_train_split", True):
            train_loader, _, layer_boundaries = data_util.get_loaders(
                self.hparams["setup_data_sample_path"],
                self.hparams["xml_path"],
                self.hparams["xml_ptype"],
                self.hparams.get("train_val_frac", 0.01),
                self.hparams.get("train_batch_size", 512),
                self.hparams["dataset_params"].get("eps", 1.0e-10),
                device="cpu",
                shuffle=bool(self.hparams.get("train_shuffle", False)),
                width_noise=self.hparams.get("width_noise", 1e-7),
                energy=self.hparams["dataset_params"].get("single_energy", None),
                u0up_cut=self.hparams["dataset_params"].get("u0up_cut", 7.0),
                u0low_cut=self.hparams["dataset_params"].get("u0low_cut", 0.0),
                rew=self.hparams["dataset_params"].get("pt_rew", 1.0),
                dep_cut=self.hparams["dataset_params"].get("dep_cut", 1.0e10),
            )
            x = train_loader.data.cpu().numpy()
            c = train_loader.cond.cpu().numpy()
        else:
            sample_data, layer_boundaries = data_util.load_data(
                self.hparams["setup_data_sample_path"],
                self.hparams["xml_ptype"],
                self.hparams["xml_path"],
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
        # Phase 2: num_train_samples is set during __init__ from data size.
        # If a Lightning datamodule is attached, prefer its count (may differ
        # if DataModule uses different preprocessing filters).
        if stage == "fit":
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
                and hasattr(self.trainer.datamodule, "num_train_samples")
            ):
                self.num_train_samples = self.trainer.datamodule.num_train_samples
                self.steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                # Fallback: keep __init__ value; steps_per_epoch computed elsewhere
                self.steps_per_epoch = max(1, self.num_train_samples // 512)
            rank_zero_info(
                f"Number of training samples: {self.num_train_samples}"
            )

    def configure_optimizers(self):
        """Build optimizer + LR scheduler, matching legacy Trainer.set_optimizer.

        Phase 4: supports all five legacy scheduler types:
        - one_cycle_lr  (used by pions_odd_discrete.yaml)
        - step
        - reduce_on_plateau
        - cycle_lr
        - multi_step_lr
        """
        opt_cfg = self.optimizer_params
        sched_cfg = self.scheduler_params
        lr_sched_mode = sched_cfg.get("lr_scheduler", "one_cycle_lr")

        # ---- Optimiser (identical to legacy) ----
        optimizer = torch.optim.AdamW(
            self.model.params_trainable,
            lr=float(opt_cfg.get("lr", 0.0002)),   # legacy default: 0.0002
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-6)),    # legacy default: 1e-6
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )

        steps_per_epoch = int(
            sched_cfg.get(
                "steps_per_epoch",
                getattr(self, "steps_per_epoch", 1),
            )
        )

        # ---- LR Scheduler (dispatched by type) ----
        if lr_sched_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("lr_decay_epochs", 30),
                gamma=sched_cfg.get("lr_decay_factor", 0.1),
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}

        elif lr_sched_mode == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.get("factor", 0.8),
                patience=sched_cfg.get("patience", 50),
                cooldown=sched_cfg.get("cooldown", 100),
                threshold=sched_cfg.get("threshold", 5e-5),
                threshold_mode=sched_cfg.get("threshold_mode", "rel"),
                verbose=True,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }

        elif lr_sched_mode == "one_cycle_lr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(
                    sched_cfg.get("max_lr", opt_cfg.get("lr", 1e-5) * 10.0)
                ),
                epochs=int(
                    sched_cfg.get("epochs", 1)
                    if "epochs" in sched_cfg
                    else sched_cfg.get("cycle_epochs", 1)
                ),
                steps_per_epoch=max(1, steps_per_epoch),
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif lr_sched_mode == "cycle_lr":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=float(sched_cfg.get("lr", opt_cfg.get("lr", 1e-5))),
                max_lr=float(sched_cfg.get("max_lr", opt_cfg.get("lr", 1e-5) * 10)),
                step_size_up=int(sched_cfg.get("step_size_up", 2000)),
                mode=sched_cfg.get("cycle_mode", "triangular"),
                cycle_momentum=False,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif lr_sched_mode == "multi_step_lr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_cfg.get(
                    "milestones", [2730, 8190, 13650, 27300]
                ),
                gamma=sched_cfg.get("gamma", 0.5),
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}

        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_sched_mode}")

        # Phase 3c: wrap scheduler to skip last 2 steps (matching legacy trainer.py:150-153)
        scheduler_config["scheduler"] = _SkipLastTwoScheduler(scheduler_config["scheduler"])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

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

    def _apply_input_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add uniform noise [0, width_noise) to input tensor.

        NOTE (Phase 3): Legacy MyDataLoader already adds noise per-batch in
        __next__, so the streaming dataset (PreprocessedStreamingDataset)
        does the same in __iter__.  The training step therefore does NOT call
        this method — adding noise here would apply it *twice*.

        This method is kept for backwards-compatibility with dataloaders
        that do not add noise (e.g., a vanilla TensorDataset).
        """
        if self.width_noise <= 0:
            return x
        return x + torch.rand_like(x) * self.width_noise

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
        # Phase 3: Noise is applied by the dataloader (matching legacy MyDataLoader).
        # _apply_input_noise is NOT called here to avoid double-noise.
        # See _apply_input_noise docstring for details.
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

        self.log("train_loss_per_dim", loss.detach() / x.shape[1], on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_inn_loss_per_dim", inn_loss.detach() / x.shape[1], on_step=True, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("train_kl_loss", kl_loss.detach(), on_step=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, c = batch
        # Phase 3: noise is in dataloader, not here
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
        self.log("val_loss_per_dim", loss.detach() / x.shape[1], on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_inn_loss_per_dim", inn_loss.detach() / x.shape[1], on_step=False, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("val_kl_loss", kl_loss.detach(), on_step=False, on_epoch=True, batch_size=x.shape[0])

    def test_step(self, batch, batch_idx):
        x, c = batch
        # Phase 3: noise is in dataloader, not here
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
        # self.log("predict_gen_time", t / c.shape[0], on_step=False, on_epoch=True, prog_bar=True, batch_size=c.shape[0])
        samples -= self.width_noise
        samples = samples[:, 0, ...]
        data = data_util.postprocess(
            samples.cpu().numpy(),
            c.cpu().numpy(),
            layer_boundaries=self.layer_boundaries,
            threshold=self.width_noise,
            quantiles=self.q.detach().cpu().numpy(),
        )

        incident_energies, layers = data_util.get_energy_and_sorted_layers(data)
        assert torch.allclose(torch.tensor(incident_energies), c.cpu(), rtol=0, atol=1e-3), "Mismatch between input energies and postprocessed energies"
        incident_energies *= 1.e3

        shower = np.concatenate(layers, axis=1) * 1.e3

        return incident_energies, shower, t / c.shape[0]


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

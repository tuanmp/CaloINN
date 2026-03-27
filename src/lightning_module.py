import math
import time

import numpy as np
import torch
import torch.distributions as dist
import pytorch_lightning as pl

import data_util
from model import CINN
import caloch_eval.evaluate as evaluate


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(
            dist.Uniform(torch.log(lb), torch.log(ub)),
            dist.ExpTransform(),
        )


class CaloINNLightningModule(pl.LightningModule):
    """Lightning module that preserves the CINN training logic."""

    def __init__(self, params, train_data, train_cond, layer_boundaries):
        super().__init__()
        self.params = params
        self.layer_boundaries = layer_boundaries
        self.single_energy = params.get("single_energy", None)
        self.num_train_samples = int(train_data.shape[0])
        self.num_dim = int(train_data.shape[1])
        self.avg_gen_time = {}
        self.width_noise = float(params.get("width_noise", 1e-7))

        data_for_init = torch.clone(train_data)
        if params.get("custom_noise", False):
            q = self.eval_quantiles(data_for_init)
        else:
            q = torch.tensor(self.width_noise, dtype=train_data.dtype)

        self.register_buffer("q", q)
        self.model = CINN(params, data_for_init, torch.clone(train_cond))

    def eval_quantiles(self, data):
        cp = torch.clone(data)
        cp[cp == 0] = torch.nan
        quantiles = torch.nanquantile(cp, q=0.01, dim=0).reshape(1, -1)
        return quantiles

    def _add_noise(self, x):
        if self.width_noise <= 0:
            return x
        return x + torch.rand_like(x) * self.width_noise

    def _compute_losses(self, x, c):
        inn_loss = -torch.mean(self.model.log_prob(x, c))
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

    def training_step(self, batch, batch_idx):
        x, c = batch
        x = self._add_noise(x)
        loss, inn_loss, kl_loss = self._compute_losses(x, c)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_inn_loss", inn_loss, on_step=True, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, c = batch
        loss, inn_loss, kl_loss = self._compute_losses(x, c)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_inn_loss", inn_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])

    def test_step(self, batch, batch_idx):
        x, c = batch
        loss, inn_loss, kl_loss = self._compute_losses(x, c)

        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("test_inn_loss", inn_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        if kl_loss is not None:
            self.log("test_kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.params_trainable,
            lr=self.params.get("lr", 0.0002),
            betas=self.params.get("betas", [0.9, 0.999]),
            eps=self.params.get("eps", 1e-6),
            weight_decay=self.params.get("weight_decay", 0.0),
        )

        lr_sched_mode = self.params.get("lr_scheduler", "reduce_on_plateau")

        if lr_sched_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim,
                step_size=self.params["lr_decay_epochs"],
                gamma=self.params["lr_decay_factor"],
            )
            return {"optimizer": optim, "lr_scheduler": scheduler}

        if lr_sched_mode == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                factor=0.8,
                patience=50,
                cooldown=100,
                threshold=5e-5,
                threshold_mode="rel",
                verbose=True,
            )
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        if lr_sched_mode == "one_cycle_lr":
            batch_size = int(self.params.get("batch_size", 512))
            drop_last = bool(self.params.get("drop_last", False))
            if drop_last:
                steps_per_epoch = self.num_train_samples // batch_size
            else:
                steps_per_epoch = math.ceil(self.num_train_samples / batch_size)
            steps_per_epoch = max(1, steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                self.params.get("max_lr", self.params["lr"] * 10),
                epochs=self.params.get("cycle_epochs") or self.params["n_epochs"],
                steps_per_epoch=steps_per_epoch,
            )
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if lr_sched_mode == "cycle_lr":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optim,
                base_lr=self.params.get("lr", 1.0e-4),
                max_lr=self.params.get("max_lr", self.params["lr"] * 10),
                step_size_up=self.params.get("step_size_up", 2000),
                mode=self.params.get("cycle_mode", "triangular"),
                cycle_momentum=False,
            )
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if lr_sched_mode == "multi_step_lr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                milestones=[2730, 8190, 13650, 27300],
                gamma=0.5,
            )
            return {"optimizer": optim, "lr_scheduler": scheduler}

        return optim

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array([
            *ret,
            *np.tile(2.0 ** 19, 5),
            *np.tile(2.0 ** 20, 3),
            *np.tile(2.0 ** 21, 2),
            *np.tile(2.0 ** 22, 1),
        ])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret

    def generate(self, num_samples, batch_size=1000, output_file=None):
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            if self.params.get("eval_dataset") == "2":
                logunif = LogUniform(torch.tensor(1e3), torch.tensor(1e6))
                energies = logunif.sample((num_samples, 1)) / 1e3
            else:
                energies = (
                    torch.tensor(
                        self.generate_Einc_ds1(
                            energy=self.single_energy,
                            sample_multiplier=1000,
                        ),
                        dtype=torch.float,
                    )
                    / 1e3
                ).reshape(-1, 1)

            samples = torch.zeros((energies.shape[0], 1, self.num_dim), device=device)
            num_samples = energies.shape[0]
            times = []

            for batch in range((num_samples + batch_size - 1) // batch_size):
                start = batch_size * batch
                stop = min(batch_size * (batch + 1), num_samples)
                energies_l = energies[start:stop].to(device)
                t1 = time.time()
                samples[start:stop] = self.model.sample(1, energies_l)
                t_diff = time.time() - t1
                times.append(t_diff / (stop - start))

            self.avg_gen_time[str(batch_size)] = np.array(times).mean()
            samples = samples[:, 0, ...].cpu().numpy()
            energies = energies.cpu().numpy()

        samples -= self.width_noise
        shower = data_util.postprocess(
            samples,
            energies,
            layer_boundaries=self.layer_boundaries,
            threshold=self.width_noise,
            quantiles=self.q.detach().cpu().numpy(),
        )

        if output_file is not None:
            save_payload = {key: np.copy(value) for key, value in shower.items()}
            data_util.save_data(save_payload, filename=output_file)

        return shower

    def plot_default_from_caloch(self, base_dir, sample_name="samples.hdf5", eval_name="final", cut=1.515e-3):
        evaluate.main(
            (
                f"-i {base_dir}/{sample_name} "
                f"-r {self.params['val_data_path']} "
                f"-m all -d {self.params['eval_dataset']} "
                f"--output_dir {base_dir}/eval/{eval_name}/ --cut {cut}"
            ).split()
        )

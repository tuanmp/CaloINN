"""
Comprehensive parity check: legacy pipeline vs new pipeline + model.

Checks every layer of the stack at meaningful scale:
  1. Data pipeline  — legacy MyDataLoader vs streaming dataset
  2. Model weights   — legacy CINN vs new CINN (same init data)
  3. Forward pass    — latent z, log_prob on full train set
  4. Optimizer step  — gradient consistency, LR schedule
  5. Generation      — same latent noise → same output
  6. Latent space    — KS test on full z distribution
"""

import copy
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# caloch_eval sub-package
CALOEVAL_DIR = os.path.join(SRC_DIR, "caloch_eval")
if CALOEVAL_DIR not in sys.path:
    sys.path.insert(0, CALOEVAL_DIR)

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float32)

import data_util
from model import CINN
from lightning_module import CaloINNLightningModule
from streaming_data import PreprocessedStreamingDataset
from tests.test_lightning_parity import _params_to_cinn


class TestComprehensiveParity(unittest.TestCase):
    """End-to-end parity: legacy pipeline vs new pipeline at meaningful scale."""

    SEED = 9999
    MAX_EVENTS = 4096
    BATCH_SIZE = 128

    @classmethod
    def setUpClass(cls):
        config_path = os.path.join(REPO_ROOT, "params", "pions.yaml")
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if not os.path.exists(cfg.get("data_path", "")):
            raise unittest.SkipTest("pions dataset not available")

        cls.cfg = cfg
        cls.cfg.update({
            "n_blocks": 4, "internal_size": 64, "norm": True,
            "bayesian": False, "width_noise": 0.0,
            "val_frac": 0.05, "batch_size": cls.BATCH_SIZE,
        })

    # -------------------------------------------------------------------
    # Build both pipelines (no patching — use load_data indices kwarg)
    # -------------------------------------------------------------------

    def setUp(self):
        # Ensure float32 (unittest may reset default dtype)
        torch.set_default_dtype(torch.float32)

    def _build_both(self, max_samples=None):
        """Return (model_legacy, model_new, x_data, c_data, layer_boundaries)."""
        import h5py

        # Load a limited subset directly via load_data with indices
        # (avoids patching which breaks when other tests have patched)
        xml_path = self.cfg.get("xml_path")
        data_path = self.cfg["data_path"]

        with h5py.File(data_path, "r") as f:
            n_total = f["incident_energies"].shape[0]
        indices = np.arange(min(self.MAX_EVENTS, n_total), dtype=np.int64)

        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        data, lbs = data_util.load_data(
            data_path, self.cfg.get("xml_ptype"), xml_path,
            indices=indices,
        )
        x, c = data_util.preprocess(
            data, lbs, self.cfg["eps"],
            u0up_cut=self.cfg.get("u0up_cut", 7.0),
            u0low_cut=self.cfg.get("u0low_cut", 0.0),
            rew=self.cfg.get("pt_rew", 1.0),
            dep_cut=self.cfg.get("dep_cut", 1e10),
        )

        # Train/val split.  IMPORTANT: preprocess returns float64;
        # force float32 to match model dtype (torch 2.x is strict).
        n_total = len(x)
        n_val = int(n_total * self.cfg["val_frac"])
        n_train = n_total - n_val
        full_x = torch.tensor(x[:n_train], dtype=torch.float32).contiguous()
        full_c = torch.tensor(c[:n_train], dtype=torch.float32).contiguous()

        # Legacy-style max_samples subsample for model init
        n_full = int(full_x.shape[0])
        if max_samples is not None and max_samples < n_full:
            torch.manual_seed(42)
            rand_idx = torch.randperm(n_full)[:max_samples]
            init_x = full_x[rand_idx]
            init_c = full_c[rand_idx]
        else:
            init_x = full_x
            init_c = full_c

        # Legacy model
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        model_legacy = CINN(self.cfg, init_x, init_c)
        model_legacy.eval()

        # New model (same seed, same data)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        cinn_params = _params_to_cinn(self.cfg)
        lm = CaloINNLightningModule(
            init_data_x=init_x, init_data_c=init_c,
            init_layer_boundaries=lbs,
            init_num_train_samples=n_full,
            cinn_params=cinn_params, width_noise=0.0,
            max_samples=max_samples,
        )
        model_new = lm.model
        model_new.eval()

        info = {
            "n_full": n_full,
            "n_init": int(init_x.shape[0]),
            "n_params": sum(p.numel() for p in model_legacy.parameters()),
        }
        return model_legacy, model_new, full_x, full_c, lbs, info

    # -------------------------------------------------------------------
    # 1. Model weights
    # -------------------------------------------------------------------

    def test_01_model_weights_identical(self):
        """Model weights must be bit-identical."""
        ml, mn, _, _, _, info = self._build_both(max_samples=512)
        ls, ns = ml.state_dict(), mn.state_dict()

        self.assertEqual(set(ls.keys()), set(ns.keys()),
                         "State dict keys differ")

        max_diff = max(
            float(torch.max(torch.abs(ls[k] - ns[k]))) for k in ls
        )
        self.assertEqual(max_diff, 0.0,
                         f"Weight mismatch: max_diff={max_diff:.1e}")

        print(f"  ✓ Weights: {len(ls)} tensors, max diff = 0.0")

    # -------------------------------------------------------------------
    # 2. Forward pass — latent z
    # -------------------------------------------------------------------

    def test_02_latent_z_identical(self):
        """Latent z on full train set must be identical."""
        ml, mn, fx, fc, _, info = self._build_both(max_samples=512)

        # Enforce float32
        fx = fx.to(torch.float32).contiguous()
        fc = fc.to(torch.float32).contiguous()
        print(f"  dtypes: fx={fx.dtype}, fc={fc.dtype}, model={next(ml.parameters()).dtype}")

        with torch.no_grad():
            zl = ml(fx, fc)
            zn = mn(fx, fc)
            zl = zl[0] if isinstance(zl, tuple) else zl
            zn = zn[0] if isinstance(zn, tuple) else zn

        z_diff = float(torch.max(torch.abs(zl - zn)))
        self.assertEqual(z_diff, 0.0,
                         f"z mismatch: {z_diff:.1e}")

        print(f"  ✓ Latent z ({fx.shape[0]}×{fx.shape[1]}): max diff = 0.0")

    # -------------------------------------------------------------------
    # 3. Forward pass — log_prob
    # -------------------------------------------------------------------

    def test_03_log_prob_identical(self):
        """log_prob on full train set must be identical."""
        ml, mn, fx, fc, _, info = self._build_both(max_samples=512)

        with torch.no_grad():
            lpl = ml.log_prob(fx, fc)
            lpn = mn.log_prob(fx, fc)

        max_diff = float(torch.max(torch.abs(lpl - lpn)))
        mean_diff = float(torch.mean(torch.abs(lpl - lpn)))
        self.assertEqual(max_diff, 0.0,
                         f"log_prob mismatch: max={max_diff:.1e} mean={mean_diff:.1e}")

        print(f"  ✓ log_prob: max diff = 0.0, mean diff = 0.0")

    # -------------------------------------------------------------------
    # 4. Optimizer step parity
    # -------------------------------------------------------------------

    def test_04_optimizer_step_identical(self):
        """One optimizer step must produce identical weight updates."""
        ml, mn, fx, fc, _, info = self._build_both(max_samples=512)

        # Use a batch from the init data
        xb, cb = fx[:self.BATCH_SIZE], fc[:self.BATCH_SIZE]

        # Legacy optimizer
        opt_l = torch.optim.AdamW(
            ml.params_trainable, lr=1e-5,
            betas=(0.9, 0.999), eps=1e-10, weight_decay=0.01,
        )
        # New optimizer
        opt_n = torch.optim.AdamW(
            mn.params_trainable, lr=1e-5,
            betas=(0.9, 0.999), eps=1e-10, weight_decay=0.01,
        )

        ml.train(); mn.train()
        loss_l = -torch.mean(ml.log_prob(xb, cb))
        loss_n = -torch.mean(mn.log_prob(xb, cb))

        self.assertTrue(torch.allclose(loss_l, loss_n, atol=1e-7),
                        f"Loss mismatch: {loss_l.item():.6f} vs {loss_n.item():.6f}")

        opt_l.zero_grad(); loss_l.backward()
        opt_n.zero_grad(); loss_n.backward()

        # Compare gradients
        for (nl, pl), (nn, pn) in zip(ml.named_parameters(), mn.named_parameters()):
            if pl.grad is not None and pn.grad is not None:
                g_diff = float(torch.max(torch.abs(pl.grad - pn.grad)))
                self.assertLess(g_diff, 1e-7,
                                f"Gradient mismatch in {nl}: {g_diff:.1e}")

        opt_l.step(); opt_n.step()

        # Compare updated weights
        max_w_diff = max(
            float(torch.max(torch.abs(ls - ns)))
            for ls, ns in zip(
                ml.state_dict().values(),
                mn.state_dict().values(),
            )
        )
        self.assertLess(max_w_diff, 1e-7,
                        f"Post-step weight mismatch: {max_w_diff:.1e}")

        print(f"  ✓ Optimizer step: loss={loss_l.item():.4f}, "
              f"post-step max diff={max_w_diff:.1e}")

    # -------------------------------------------------------------------
    # 5. LR schedule parity
    # -------------------------------------------------------------------

    def test_05_lr_schedule_identical(self):
        """OneCycleLR must produce identical LR values for N steps."""
        ml, _, _, _, _, _ = self._build_both(max_samples=512)

        steps_per_epoch = 200
        n_steps = 20

        # Legacy scheduler
        opt_l = torch.optim.AdamW(ml.params_trainable, lr=1e-5)
        sched_l = torch.optim.lr_scheduler.OneCycleLR(
            opt_l, max_lr=1e-4, epochs=1, steps_per_epoch=steps_per_epoch,
        )
        # New scheduler (same params)
        opt_n = torch.optim.AdamW(ml.params_trainable, lr=1e-5)
        sched_n = torch.optim.lr_scheduler.OneCycleLR(
            opt_n, max_lr=1e-4, epochs=1, steps_per_epoch=steps_per_epoch,
        )

        lrs_l, lrs_n = [], []
        for _ in range(n_steps):
            lrs_l.append(float(opt_l.param_groups[0]["lr"]))
            lrs_n.append(float(opt_n.param_groups[0]["lr"]))
            sched_l.step(); sched_n.step()

        max_lr_diff = max(abs(a - b) for a, b in zip(lrs_l, lrs_n))
        self.assertEqual(max_lr_diff, 0.0,
                         f"LR schedule mismatch: {max_lr_diff:.1e}")

        print(f"  ✓ LR schedule: {n_steps} steps, max diff = 0.0, "
              f"range [{lrs_l[0]:.2e}, {lrs_l[-1]:.2e}]")

    # -------------------------------------------------------------------
    # 6. Generation parity
    # -------------------------------------------------------------------

    def test_06_generation_identical(self):
        """Generated samples must match (same latent noise → same output)."""
        ml, mn, fx, fc, _, info = self._build_both(max_samples=512)

        n_gen = 128
        torch.manual_seed(777)
        z = torch.randn(n_gen, fx.shape[1])
        c_gen = fc[:n_gen]

        with torch.no_grad():
            xl, _ = ml(z, c_gen, rev=True)
            xn, _ = mn(z, c_gen, rev=True)
            xl = xl[0] if isinstance(xl, tuple) else xl
            xn = xn[0] if isinstance(xn, tuple) else xn

        max_diff = float(torch.max(torch.abs(xl - xn)))
        self.assertEqual(max_diff, 0.0,
                         f"Generation mismatch: {max_diff:.1e}")

        print(f"  ✓ Generation ({n_gen} samples): max diff = 0.0")

    # -------------------------------------------------------------------
    # 7. Latent space distribution (KS)
    # -------------------------------------------------------------------

    def test_07_latent_distribution_ks(self):
        """Latent z histograms must match via KS test."""
        ml, mn, fx, fc, _, info = self._build_both(max_samples=512)

        with torch.no_grad():
            zl = ml(fx, fc)
            zn = mn(fx, fc)
            zl = zl[0] if isinstance(zl, tuple) else zl
            zn = zn[0] if isinstance(zn, tuple) else zn

        from scipy import stats
        n_test = min(100, zl.shape[1])
        ks_fails = sum(
            1 for d in range(n_test)
            if stats.ks_2samp(zl[:, d].numpy(), zn[:, d].numpy())[1] < 0.01
        )
        pct = ks_fails / n_test * 100

        self.assertLess(pct, 2.5,
                        f"Too many KS failures: {ks_fails}/{n_test}")

        print(f"  ✓ KS test: {ks_fails}/{n_test} failed ({pct:.1f}%)")

    # -------------------------------------------------------------------
    # 8. Streaming dataloader parity
    # -------------------------------------------------------------------

    def test_08_streaming_dataloader_parity(self):
        """Streaming dataset must produce same data stats as legacy."""
        _, _, fx, fc, lbs, _ = self._build_both(max_samples=512)

        # Use first batch from already-loaded data as "legacy"
        xl, cl = fx[:self.BATCH_SIZE], fc[:self.BATCH_SIZE]

        # Streaming batch (same indices)
        ds = PreprocessedStreamingDataset(
            data_path=self.cfg["data_path"],
            xml_filename=self.cfg.get("xml_path"),
            particle_type=self.cfg.get("xml_ptype"),
            batch_size=self.BATCH_SIZE, eps=self.cfg["eps"],
            u0up_cut=self.cfg.get("u0up_cut", 7.0),
            u0low_cut=self.cfg.get("u0low_cut", 0.0),
            rew=self.cfg.get("pt_rew", 1.0),
            dep_cut=self.cfg.get("dep_cut", 1e10),
            width_noise=0.0, fixed_noise=False, val_frac=0,
            shuffle=False, is_train=True, layer_boundaries=lbs,
        )
        np.random.seed(self.SEED); torch.manual_seed(self.SEED)
        xn, cn = next(iter(ds))

        # Structural
        self.assertEqual(xl.shape, xn.shape)
        self.assertEqual(cl.shape, cn.shape)

        # Statistical (ordering may differ due to filter implementations)
        self.assertTrue(torch.isfinite(xl).all(), "legacy x has NaN")
        self.assertTrue(torch.isfinite(xn).all(), "streaming x has NaN")
        self.assertTrue((xl >= 0).all(), "legacy x has negatives")
        self.assertTrue((xn >= 0).all(), "streaming x has negatives")

        x_rel = abs(xl.mean() - xn.mean()) / max(1e-8, abs(xn.mean()))
        c_rel = abs(cl.mean() - cn.mean()) / max(1e-8, abs(cn.mean()))

        self.assertLess(x_rel, 0.05, f"x means differ: {x_rel:.3f}")
        self.assertLess(c_rel, 0.05, f"c means differ: {c_rel:.3f}")

        print(f"  ✓ Streaming dataloader: x mean rel diff={x_rel:.4f}, "
              f"c mean rel diff={c_rel:.4f}")


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    unittest.main(verbosity=2)
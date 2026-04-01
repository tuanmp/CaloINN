#!/usr/bin/env python3
import argparse
import copy
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
from lightning_data import CaloINNDataModule
from lightning_module import CaloINNLightningModule
from trainer import Trainer


@dataclass
class SeedResult:
    seed: int
    n_steps: int
    old_mean: float
    old_std: float
    new_mean: float
    new_std: float
    rel_mean_diff: float
    std_ratio: float


class DummyDoc:
    def __init__(self, basedir: str):
        self.basedir = basedir

    def get_file(self, name: str, add_run_name: bool = False) -> str:
        return os.path.join(self.basedir, name)


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def force_actnorm_pending(cinn_model) -> int:
    count = 0
    for mod in cinn_model.modules():
        if hasattr(mod, "init_on_next_batch"):
            mod.init_on_next_batch = True
            count += 1
    return count


def patch_load_data(max_events: int):
    original = data_util.load_data

    def _patched(*args, **kwargs):
        data, layer_boundaries = original(*args, **kwargs)
        sliced = {k: v[:max_events] for k, v in data.items()}
        return sliced, layer_boundaries

    data_util.load_data = _patched
    return original


def restore_load_data(original):
    data_util.load_data = original


def build_lightning_components(light_cfg: Dict) -> Tuple[CaloINNDataModule, CaloINNLightningModule]:
    dm = CaloINNDataModule(
        data_path=light_cfg["data"]["data_path"],
        val_data_path=light_cfg["data"]["val_data_path"],
        batch_size=light_cfg["data"]["batch_size"],
        cond_key=light_cfg["data"].get("cond_key", "incident_energies"),
        sample_key=light_cfg["data"].get("sample_key", "showers"),
        val_frac=light_cfg["data"].get("val_frac", 0.01),
        shuffle=bool(light_cfg["data"].get("shuffle", False)),
        eval_dataset=light_cfg["data"].get("eval_dataset", "1-pions"),
        num_workers=int(light_cfg["data"].get("num_workers", 0)),
        predict_batch_size=light_cfg["data"].get("predict_batch_size", light_cfg["data"]["batch_size"]),
        dataset_kwargs=copy.deepcopy(light_cfg["data"]["dataset_kwargs"]),
    )
    dm.setup("fit")

    model_cfg = light_cfg["model"]
    module = CaloINNLightningModule(
        setup_data_sample_path=model_cfg["setup_data_sample_path"],
        enable_diagnostics=bool(model_cfg.get("enable_diagnostics", False)),
        actnorm_calibration_samples=int(model_cfg.get("actnorm_calibration_samples", 1024)),
        xml_path=model_cfg["xml_path"],
        xml_ptype=model_cfg["xml_ptype"],
        dataset_params=copy.deepcopy(model_cfg["dataset_params"]),
        cinn_params=copy.deepcopy(model_cfg["cinn_params"]),
        width_noise=float(model_cfg.get("width_noise", 0.0)),
        custom_noise=bool(model_cfg.get("custom_noise", False)),
        single_energy=model_cfg.get("single_energy", None),
    ).to("cpu")
    return dm, module


def make_optim_scheduler_old(legacy: Trainer, cfg: Dict, steps: int):
    opt = torch.optim.AdamW(
        legacy.model.params_trainable,
        lr=float(cfg.get("lr", 1e-5)),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        eps=float(cfg.get("eps", 1e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(cfg.get("max_lr", cfg.get("lr", 1e-5) * 10.0)),
        epochs=int(cfg.get("cycle_epochs", cfg.get("n_epochs", 1))),
        steps_per_epoch=max(1, steps),
    )
    return opt, sched


def make_optim_scheduler_new(module: CaloINNLightningModule, cfg: Dict, steps: int):
    opt_cfg = cfg["optimizer"]["init_args"]
    sched_cfg = cfg["lr_scheduler"]["init_args"]

    opt = torch.optim.AdamW(
        module.model.params_trainable,
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(sched_cfg["max_lr"]),
        epochs=int(sched_cfg.get("epochs", 1)),
        steps_per_epoch=max(1, steps),
    )
    return opt, sched


def run_seed(seed: int, legacy_cfg: Dict, light_cfg: Dict, n_steps: int, max_events: int, force_pending: bool) -> SeedResult:
    set_seed(seed)
    original_load_data = patch_load_data(max_events=max_events)
    tempdir = tempfile.mkdtemp(prefix=f"parity_seed_{seed}_")

    try:
        legacy = Trainer(copy.deepcopy(legacy_cfg), "cpu", DummyDoc(tempdir))
        dm, module = build_lightning_components(light_cfg)

        module.model.load_state_dict(copy.deepcopy(legacy.model.state_dict()))
        if force_pending:
            force_actnorm_pending(module.model)

        opt_old, sched_old = make_optim_scheduler_old(legacy, legacy_cfg, n_steps)
        opt_new, sched_new = make_optim_scheduler_new(module, light_cfg, n_steps)

        old_iter = iter(legacy.train_loader)
        new_iter = iter(dm.train_dataloader())

        losses_old: List[float] = []
        losses_new: List[float] = []

        for _ in range(n_steps):
            try:
                x_old, c_old = next(old_iter)
                x_new, c_new = next(new_iter)
            except StopIteration:
                break

            legacy.model.train()
            module.model.train()

            opt_old.zero_grad()
            loss_old = -torch.mean(legacy.model.log_prob(x_old, c_old))
            loss_old.backward()
            opt_old.step()
            sched_old.step()

            opt_new.zero_grad()
            loss_new, _, _ = module._compute_losses(x_new, c_new)
            loss_new.backward()
            opt_new.step()
            sched_new.step()

            losses_old.append(float(loss_old.detach().cpu().item()))
            losses_new.append(float(loss_new.detach().cpu().item()))

        if not losses_old or not losses_new:
            raise RuntimeError("No training steps were executed. Increase max_events or decrease n_steps.")

        old_mean = float(np.mean(losses_old))
        old_std = float(np.std(losses_old))
        new_mean = float(np.mean(losses_new))
        new_std = float(np.std(losses_new))

        rel_mean_diff = abs(new_mean - old_mean) / max(abs(old_mean), 1e-12)
        std_ratio = new_std / max(old_std, 1e-12)

        return SeedResult(
            seed=seed,
            n_steps=len(losses_old),
            old_mean=old_mean,
            old_std=old_std,
            new_mean=new_mean,
            new_std=new_std,
            rel_mean_diff=rel_mean_diff,
            std_ratio=std_ratio,
        )
    finally:
        restore_load_data(original_load_data)
        shutil.rmtree(tempdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Statistical parity benchmark between legacy and Lightning training paths.")
    parser.add_argument("--legacy-config", default=os.path.join(ROOT, "params", "pions.yaml"))
    parser.add_argument("--lightning-config", default=os.path.join(ROOT, "params", "lightning_pion.yaml"))
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=2048)
    parser.add_argument("--tol-mean-rel", type=float, default=0.05)
    parser.add_argument("--std-ratio-min", type=float, default=0.8)
    parser.add_argument("--std-ratio-max", type=float, default=1.25)
    parser.add_argument("--no-force-actnorm-pending", action="store_true")
    args = parser.parse_args()

    legacy_cfg = load_yaml(args.legacy_config)
    light_cfg = load_yaml(args.lightning_config)

    # Keep critical knobs aligned for a fair statistical benchmark.
    light_cfg = copy.deepcopy(light_cfg)
    light_cfg["data"]["dataset_kwargs"]["width_noise"] = float(legacy_cfg.get("width_noise", 0.0))
    light_cfg["model"]["width_noise"] = float(legacy_cfg.get("width_noise", 0.0))
    light_cfg["optimizer"]["init_args"]["eps"] = float(legacy_cfg.get("eps", 1e-6))
    light_cfg["lr_scheduler"]["init_args"]["epochs"] = int(legacy_cfg.get("cycle_epochs", legacy_cfg.get("n_epochs", 1)))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    force_pending = not args.no_force_actnorm_pending

    print("Statistical Parity Benchmark")
    print(f"legacy_config={args.legacy_config}")
    print(f"lightning_config={args.lightning_config}")
    print(f"seeds={seeds}")
    print(f"steps={args.steps}, max_events={args.max_events}")
    print(f"force_actnorm_pending={force_pending}")
    print(
        f"thresholds: rel_mean_diff<={args.tol_mean_rel}, "
        f"std_ratio in [{args.std_ratio_min}, {args.std_ratio_max}]"
    )
    print("-")

    results: List[SeedResult] = []
    for seed in seeds:
        res = run_seed(
            seed=seed,
            legacy_cfg=legacy_cfg,
            light_cfg=light_cfg,
            n_steps=args.steps,
            max_events=args.max_events,
            force_pending=force_pending,
        )
        results.append(res)
        print(
            "seed={seed} steps={steps} old_mean={old_mean:.6f} new_mean={new_mean:.6f} "
            "rel_mean_diff={rel_mean_diff:.6%} old_std={old_std:.6f} new_std={new_std:.6f} std_ratio={std_ratio:.6f}".format(
                seed=res.seed,
                steps=res.n_steps,
                old_mean=res.old_mean,
                new_mean=res.new_mean,
                rel_mean_diff=res.rel_mean_diff,
                old_std=res.old_std,
                new_std=res.new_std,
                std_ratio=res.std_ratio,
            )
        )

    rel_diffs = np.array([r.rel_mean_diff for r in results], dtype=float)
    std_ratios = np.array([r.std_ratio for r in results], dtype=float)

    rel_ok = bool(np.all(rel_diffs <= args.tol_mean_rel))
    std_ok = bool(np.all((std_ratios >= args.std_ratio_min) & (std_ratios <= args.std_ratio_max)))
    all_ok = rel_ok and std_ok

    print("-")
    print(
        "aggregate: rel_mean_diff mean={:.6%}, max={:.6%}; std_ratio mean={:.6f}, min={:.6f}, max={:.6f}".format(
            float(np.mean(rel_diffs)),
            float(np.max(rel_diffs)),
            float(np.mean(std_ratios)),
            float(np.min(std_ratios)),
            float(np.max(std_ratios)),
        )
    )
    print(f"PASS={all_ok} (rel_ok={rel_ok}, std_ok={std_ok})")

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

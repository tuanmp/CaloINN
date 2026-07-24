# AGENTS.md — CaloINN Lightning

## Environment & Package Management

**Always use `uv` for everything.** Python 3.11. `pyproject.toml` is the sole source of truth for dependencies. Ignore `requirements.txt` (stale legacy torch 1.13 pins).

## Required Environment Variable

`HDF5_USE_FILE_LOCKING=FALSE` is **mandatory** for any command touching HDF5. Pre-approved in `opencode.json`.

## Entrypoints

| Purpose | Command |
|---------|---------|
| **Train** | `uv run python main.py fit --config params/pions_odd.yaml` |
| **Test (all)** | `HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/ -v` |
| **Test (single file)** | `HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_lightning_data.py -v` |
| **Test MCMC unit tests** | `HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_mcmc.py -v -k "not gpu"` |

- `./main.py` = LightningCLI entrypoint (sole CLI — no legacy CLI exists)

## Architecture

```
main.py (LightningCLI)
  └── trainer_class = training_utils.trainer.Trainer (SLURM-aware)
  └── model = CaloINNLightningModule (src/lightning_module.py)
  │     └── CINN model (src/model.py)
  └── data  = CaloINNDataModule (src/lightning_data.py)
        ├── use_sharded=true  → src/sharded_data.py (prod, num_workers>0)
        └── else              → src/streaming_data.py (IterableDataset, num_workers=0)
```

**Key source files:**
- `src/model.py` — core CINN (conditional invertible NN via FrEIA), do not modify
- `src/lightning_module.py` — Lightning wrapper: `training_step`, `configure_optimizers`, `conditional_generate`, `predict_step`, `mcmc_sample`
- `src/lightning_data.py` — `CaloINNDataModule` (dispatches to streaming or sharded backend)
- `src/sharded_data.py` — production map-style Dataset with HDF5 caching (`cache_mode: memmap`)
- `src/streaming_data.py` — HDF5 streaming IterableDataset (~57 MB RAM)
- `src/data_util.py` — preprocessing/postprocessing (normalize layers, extra dims, energy validation)
- `training_utils/trainer.py` — custom Lightning `Trainer` with SLURM-aware `default_root_dir`
- `src/myBlocks/` — custom FrEIA subnet blocks (MADE, cubic, spline)
- `src/splines/` — rational quadratic and linear spline implementations
- `src/mcmc/` — MCMC density-ratio correction (classifier, calibration, IMH sampler)
- `src/caloch_eval/` — CaloChallenge evaluation pipeline (HighLevelFeatures, plotting)
- `src/vblinear.py` — variational Bayes Linear layer for Bayesian training

## Data Preprocessing (Critical)

Uses `use_extra_dims` mode (default):
1. Raw HDF5 showers (MeV) → divided by 1e3 to GeV
2. Each layer normalized by its own energy (`normalize_layers`)
3. Extra dimensions appended: total energy / incident energy, plus per-layer energy fractions
4. Input shape = shower_cells + num_layers (e.g., 504 + 36 = 540 for odd pion)

Postprocessing reverses: `unnormalize_layers` reconstructs layer energies from extra dims, then multiplies back.

Sharded mode (`use_sharded: true`) supports caching via `cache_mode: memmap` + `cache_dir` to skip repeated preprocessing.

## Tests

- `tests/test_lightning_data.py` — CaloINNDataModule and HDF5IterableDataset tests
- `tests/test_streaming_parity.py` — LegacyStreamingDataModule parity and shape tests
- `tests/test_mcmc.py` — MCMC density-ratio correction tests (GPU tests need checkpoints)

Tests require the dataset at the path in `params/pions.yaml` (`data_path`). Auto-skip if unavailable.

## Key Gotchas

1. **`torch.set_default_dtype(torch.float32)` before everything** — CINN model requires float32. `main.py` does this at module level before any imports.

2. **No double-noise** — `training_step` does NOT add noise. Noise is applied by the dataloader (both streaming and sharded paths). Adding noise in training_step applies it twice.

3. **`num_workers` and HDF5** — Streaming (`use_sharded: false`) requires `num_workers: 0`. Sharded (`use_sharded: true`) supports `num_workers > 0` via pickle-safe file handles and per-worker HDF5 handles.

4. **`_SkipLastTwoScheduler`** wraps the LR scheduler to skip the last 2 `step()` calls (legacy behavior — prevents OneCycleLR terminal annealing).

5. **`num_train_samples`** critical for Bayesian models (KL loss scaling). Set from actual dataset size during init, not hardcoded.

6. **FrEIA from PyPI** — `freia>=0.2` (not a custom install). Uses `FrEIA.framework` + `FrEIA.modules` (note capital F).

7. **LightningCLI config** — params use YAML with `class_path`/`init_args`. `params/pions_odd.yaml` is the canonical example.

8. **SLURM-aware Trainer** — `training_utils/trainer.py` auto-detects SLURM and uses `$SLURM_JOB_ID` as the run directory. In interactive mode, uses timestamp.

9. **`predict_step` sampling pipeline** — generates via `conditional_generate`, subtracts `width_noise`, runs `postprocess`, validates energies, returns `(incident_energies_MeV, showers_MeV, gen_time)`.

10. **Sharded data caching** — `cache_mode: memmap` with `cache_dir` persists preprocessed data to disk, avoiding re-preprocessing across runs. The cache is keyed by data path + preprocessing params hash.

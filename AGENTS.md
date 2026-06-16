# AGENTS.md — CaloINN Lightning

## Environment & Package Management

**Always use `uv` for everything:** install, run, test. The repo uses `pyproject.toml` as the sole source of truth for dependencies. `requirements.txt` is stale (legacy torch 1.13 pins) — ignore it.

```bash
# Python 3.11
uv run python ...
uv run python -m pytest tests/ -v
```

## Required Environment Variable

`HDF5_USE_FILE_LOCKING=FALSE` is **mandatory** for any command touching HDF5. The `opencode.json` already grants pre-approval for `HDF5_USE_FILE_LOCKING=FALSE uv *`.

## Entrypoints

| Purpose | Command |
|---------|---------|
| **Train (Lightning)** | `uv run python main.py fit --config params/pions_odd.yaml` |
| **Test (all)** | `HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/ -v` |
| **Test (single file)** | `HDF5_USE_FILE_LOCKING=FALSE uv run python tests/test_lightning_parity.py -v` |
| **Legacy train** | `python src/main.py params/<card>.yaml -c` |

- `./main.py` = LightningCLI entrypoint (current)
- `src/main.py` = legacy CLI (deprecated, kept for parity reference)

## Architecture

```
main.py (LightningCLI)
  ├── CaloINNDataModule (lightning_data.py)
  │     ├── use_sharded=true → ShardedDataModule (sharded_data.py)
  │     └── use_sharded=false → LegacyStreamingDataModule (streaming_data.py)
  └── CaloINNLightningModule (lightning_module.py)
        └── CINN model (model.py) — unchanged from legacy
```

**Key files:**
- `src/model.py` — core CINN (conditional invertible neural network), unchanging
- `src/lightning_module.py` — Lightning wrapper: training_step, configure_optimizers, conditional_generate, predict_step
- `src/lightning_data.py` — CaloINNDataModule with streaming/sharded backends
- `src/streaming_data.py` — HDF5 streaming dataset (~57 MB RAM, replaces legacy 7 GB MyDataLoader)
- `src/data_util.py` — preprocessing/postprocessing (normalize layers, extra dims, energy validation)

**Legacy reference (do NOT modify unless parity testing):**
- `CaloINN/src/trainer.py` — original Trainer (reference for parity tests)
- `src/trainer.py` — copy of legacy trainer
- `src/myDataLoader.py` — legacy in-memory dataloader
- `src/main.py` — legacy CLI
- `src/lightning_runner.py` — old Lightning wrapper, uses outdated API signatures

## Data Preprocessing (Critical)

The data pipeline uses `use_extra_dims` mode (default):
1. Raw HDF5 showers (MeV) → divided by 1e3 (GeV)
2. Each layer normalized by its own energy (`normalize_layers`)
3. Extra dimensions appended: total energy / incident energy, plus per-layer energy fractions
4. Input shape = shower_cells + num_layers (e.g., 504 + 36 = 540 for odd pion)

Postprocessing reverses this: `unnormalize_layers` reconstructs layer energies from extra dims, then multiplies back.

## Tests

Tests are **parity tests** — they validate the Lightning refactor produces bit-identical results to the legacy pipeline. They use `unittest` (run with pytest or unittest directly).

Tests require the dataset at the path specified in `params/pions.yaml` (`data_path`). If not available, tests auto-skip.

**Important:** `torch.set_default_dtype(torch.float32)` must be set before model creation — `main.py` and all test files do this explicitly.

## Key Gotchas

1. **`torch.set_default_dtype(torch.float32)` before everything** — the CINN model requires float32. Must be called at module level or before any tensor creation.

2. **No double-noise**: `training_step` does NOT add noise. Noise is applied by the dataloader (matching legacy `MyDataLoader.__next__`). Adding noise in training_step would apply it twice.

3. **`num_workers` and HDF5**: Streaming uses `num_workers: 0` (HDF5 not fork-safe). Sharded (`use_sharded: true`) supports `num_workers > 0` via pickle-safe file handles.

4. **`_SkipLastTwoScheduler`** wraps the LR scheduler to skip the last 2 steps (matching legacy `trainer.py` lines 150-153). This prevents OneCycleLR from reaching its terminal annealing.

5. **`num_train_samples`** is critical for Bayesian models (KL loss scaling). It's set from the actual dataset size during init, not hardcoded to 1.

6. **FrEIA**: The `frEIA==0.2` package from PyPI (not a custom install). Provides the invertible network framework.

7. **Config is LightningCLI-based**: Params use YAML with `class_path`/`init_args` structure. See `params/pions_odd.yaml` as the canonical example.

8. **Slurm-aware default root dir**: The custom `Trainer` class (in `training_utils/trainer.py`) auto-detects SLURM and uses `$SLURM_JOB_ID` as the run directory name.

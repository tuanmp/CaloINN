# Legacy Pipeline & Trainer Removal

**Date:** 2026-06-16
**Status:** Approved

## Goal

Remove the legacy training pipeline (pre-Lightning refactor) and all code that exists only to validate parity between legacy and Lightning. The Lightning pipeline is now the ground truth.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Remove all legacy-vs-Lightning parity tests | They served their purpose; Lightning is now the reference |
| Refactor `get_loaders()` to use streaming | Consistency — no inline `MyDataLoader` copy; use the modern pipeline |
| Keep `src/plotting.py` | Still used by `training_utils/latent_sample.py` |
| Point configs to root `binning_pion_odd_coarse_hcal.xml` | Byte-identical to `CaloINN/binning_odd_pion.xml`; already at root |

## Files to Remove

### Core Legacy Source (`src/`)
```
src/trainer.py          — Legacy Trainer class
src/main.py             — Legacy CLI entrypoint  
src/myDataLoader.py     — Legacy in-memory dataloader
src/lightning_runner.py — Old Lightning wrapper
src/documenter.py       — Legacy documenter
src/gen_data.py         — Standalone generation script
src/plotter.py          — Only used by legacy trainer
src/plotting2d.py       — Standalone plotting script
src/plotting_caloch.py  — Standalone plotting script
```

### Git Submodule
```
CaloINN/                — Entire legacy reference codebase
```

### Parity Tests
```
tests/test_lightning_parity.py      — 5 test classes, all legacy-vs-Lightning
tests/test_final_parity.py          — Legacy-vs-Lightning
tests/test_sharded_parity.py        — Legacy-vs-Lightning  
tests/test_comprehensive_parity.py  — 6-layer legacy-vs-Lightning stack
```

### Ancillary
```
scripts/benchmark_statistical_parity.py  — Legacy-vs-Lightning benchmark
params/pions.yaml                        — Legacy flat-format config
params/example.yaml                      — Legacy example
params/example_bayesian.yaml             — Legacy example
params/eplus.yaml                        — Legacy config
params/diagnostic.yaml                   — Legacy config
diagnostic_deep_cinn.py                  — Standalone diagnostic
diagnostic_nan.py                        — Standalone diagnostic
diagnostic_simple.py                     — Standalone diagnostic
REFACTOR_PLAN.md                         — Historical doc
RESTRUCTURE_COMPLETE.md                  — Historical doc
planning/                                — Historical docs
requirements.txt                         — Stale (pyproject.toml is source of truth)
results/                                 — Legacy training outputs
lightning_logs/                          — Old Lightning logs
slurm_logs/                              — Old SLURM logs
notebooks/                               — Legacy analysis notebooks
```

## Files to Refactor

### `src/data_util.py` — `get_loaders()` function

**Current state:** Creates `MyDataLoader` instances using in-memory preprocessed tensors.

**Target state:** Replace with `PreprocessedStreamingDataset` + PyTorch `DataLoader`. The streaming dataset already implements identical preprocessing logic. Keep the function signature and return type stable so callers don't change.

**Callers (after removal):**
- `src/lightning_module.py:148` — `load_init_tensors()` — small sample for model init
- `src/lightning_module.py:573` — `generate_latent()` — encodes validation showers

### Config XML Path Fixes

After removing `CaloINN/`, update `xml_path` references to use `binning_pion_odd_coarse_hcal.xml` at repo root (byte-identical to `CaloINN/binning_odd_pion.xml`):

| File | Lines | Change |
|------|-------|--------|
| `params/pions_odd.yaml` | 63, 83 | `.../CaloINN/binning_odd_pion.xml` → `binning_pion_odd_coarse_hcal.xml` |
| `params/pions_odd_stream.yaml` | 62, 81 | Same |
| `params/pions_odd_sharded.yaml` | 64, 84 | Same |
| `tests/test_streaming_parity.py` | 38 | Fix `CaloINN/binning_odd_pion.xml` → root path; remove `CaloINN/` from `sys.path` |

## Files That Stay

| File | Role |
|------|------|
| `src/model.py` | Core CINN model |
| `src/lightning_module.py` | Lightning wrapper (callers of `get_loaders()`) |
| `src/lightning_data.py` | CaloINNDataModule router |
| `src/streaming_data.py` | HDF5 streaming pipeline |
| `src/sharded_data.py` | Sharded map-style pipeline |
| `src/data_util.py` | Preprocessing/postprocessing (refactored) |
| `src/plotting.py` | Used by `training_utils/latent_sample.py` |
| `src/training_diagnostics.py` | Lightning diagnostics |
| `src/vblinear.py` | Model sublayer |
| `src/XMLHandler.py` | XML binning parser |
| `src/myBlocks/` | Network building blocks |
| `src/splines/` | Spline implementations |
| `src/caloch_eval/` | Physics evaluation |
| `training_utils/` | Lightning utilities (trainer, callbacks, wandb) |
| `tests/test_streaming_parity.py` | Streaming module tests (not legacy-vs-Lightning) |
| `tests/test_lightning_data.py` | Lightning data pipeline tests |
| `main.py` | LightningCLI entrypoint |
| All `params/pions_odd*.yaml`, `params/full_train_*.yaml` | Active configs |
| `pyproject.toml` | Package config |
| `batch/`, `plot_params/` | Slurm scripts, plot configs |

## Execution Order

1. Refactor `data_util.get_loaders()` to use streaming backend (unblocked by removals)
2. Update config XML paths and test references
3. Remove `CaloINN/` git submodule (`git rm`, `.gitmodules` cleanup)
4. Remove all files listed in "Files to Remove"
5. Run remaining tests (`test_lightning_data.py`, `test_streaming_parity.py`) to verify nothing is broken

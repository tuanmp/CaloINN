# Refactor Plan: CaloINN Lightning Upgrade with Parity Preservation

## Context

This plan addresses the systematic upgrade of the CaloINN codebase from a legacy PyTorch 1.x / custom Trainer architecture to PyTorch 2.x / Lightning 2.6.1, while preserving bit-for-bit reproducibility of training results.

**Reference config**: `CaloINN/params/pions_odd_discrete.yaml`
**Training data**: `/pscratch/sd/p/pmtuan/ddsim/single_pion_discrete_coarse_hcal/trainset_flat_good_order.hdf5` (2.39M events, 720 features)
**Validation data**: `/pscratch/sd/p/pmtuan/ddsim/single_pion_discrete_clf_train_coarse_hcal/all_showers_flatten.h5`

---

## Part 1: Root Cause Analysis

### Legacy Code Problems

| # | Problem | Location | Impact |
|---|---------|----------|--------|
| 1 | **Memory-inefficient dataloader** | `src/myDataLoader.py` stores entire dataset in RAM as `self.data` + `self.cond` tensors. With `fixed_noise=True` stores a second noised copy. For 2.4M events × 720 features × 4 bytes = ~6.9 GB just for data, plus copies. | Crashes on large datasets, cannot scale |
| 2 | **Custom Trainer with no checkpointing** | `src/trainer.py` manually implements save/load with custom file naming. No Lightning integration = no fault-tolerant training, no auto-resume, no HPC compatibility. | Cannot train on HPC clusters reliably |
| 3 | **PyTorch 1.x lock** | `pyproject.toml` pins `torch==1.13`. FrEIA library is written for torch 1.x and not updated. | Blocks upgrade to torch 2.x / lightning 2.6.1 |

### New Code Divergence Sources

| # | Problem | Location | Severity | Description |
|---|---------|----------|----------|-------------|
| 1 | **Still uses TensorDataset** | `src/lightning_data.py:104` uses `TensorDataset(x_trn, c_trn)` which loads entire preprocessed arrays into RAM | **Critical** | Same memory problem as legacy; defeats purpose of refactor |
| 2 | **`num_train_samples` hardcoded to 1** | `src/lightning_module.py:71` sets `self.num_train_samples = 1`. The KL scaling in `_compute_losses()` uses `self.num_train_samples / N` which produces wrong scaling until `setup()` fixes it | **High** | Causes KL loss to be scaled incorrectly (by factor of ~120k) |
| 3 | **Model initialization path differs** | Legacy: `Trainer.__init__` → `CINN(params, data, cond)` with data already in memory. New: `__init__` → `load_init_tensors()` → `get_loaders()` → extracts numpy → creates CINN | **Medium** | Different initialization order may cause subtle differences in ActNorm initialization |
| 4 | **Noise application differs** | Old `MyDataLoader` adds noise per-batch with `torch.rand_like(x) * width_noise` using `torch.randperm` index shuffle. Lightning applies noise in `training_step` via `_apply_input_noise(x) = x + torch.rand_like(x) * width_noise`. Different random states. | **High** | With same seed, old and new produce different noisy inputs |
| 5 | **`LightningRunner` uses outdated API** | `src/lightning_runner.py` passes `params` dict directly to `CaloINNDataModule(params)` and `CaloINNLightningModule(params, ...)` but these classes now take individual arguments | **High** | `src/main.py --lightning` fails with current code |
| 6 | **Scheduler configuration mismatch** | Legacy supports: `step`, `reduce_on_plateau`, `one_cycle_lr`, `cycle_lr`, `multi_step_lr`. Lightning module only supports `OneCycleLR` | **Medium** | Different scheduler = different LR per step → diverged training |

---

## Part 2: Phase Plan

### Phase 1: Memory-Efficient HDF5 Iterable Dataset

**Goal**: Replace `TensorDataset` with streaming dataset that never loads full data into RAM.

**New file**: `src/streaming_data.py`

**Implementation**:
```python
class HDF5IterableDataset(torch.utils.data.IterableDataset):
    """
    Memory-efficient streaming dataset for HDF5 calorimeter data.
    Applies preprocessing (normalize, filter, add noise) on-the-fly per batch.
    Never loads full dataset into RAM.
    """
```

**Key features**:
- Lazy HDF5 file opening (only metadata read at init)
- Per-batch preprocessing via worker function
- Optional noise addition per batch (configurable seed per epoch)
- Epoch-based shuffling via `torch.randperm` index (same as old `MyDataLoader`)
- Support for train/val split by index range

**Changes to `src/lightning_data.py`**:
- Add `HDF5IterableDataset` to `CaloINNDataModule`
- Replace `TensorDataset` with `HDF5IterableDataset` for training
- Maintain identical train/val split logic as legacy `get_loaders()`
- Ensure `num_train_samples` is computed correctly from dataset length

**Verification**:
- Memory usage stays constant regardless of dataset size (~57MB for 2.4M events)
- First batch matches exactly when noise is disabled (max diff = 0.0)
- Progress bars added for long index-computation loops (~30s, now visible)

**Discovery**: `lightning_module.py` calls `_apply_input_noise` in `training_step` while
`streaming_data.py` also adds noise in the dataset's `__iter__`. This results in **double
noise** when both are used together. To be fixed in Phase 3.

---

### Phase 2: Fix CaloINNLightningModule Initialization Parity

**Goal**: Ensure model initialization produces identical weights regardless of initialization path.

**Changes to `src/lightning_module.py`**:
1. Add `load_init_tensors_from_arrays(x, c, layer_boundaries)` method that accepts pre-processed arrays directly (bypassing file re-read)
2. Ensure ActNorm layers are calibrated identically to legacy approach
3. Fix `num_train_samples` initialization: set from `len(train_loader.data)` in `__init__` after model creation, not hardcoded to 1
4. Add `init_from_existing_data` flag to skip redundant data loading when called from Trainer context

**Verification**:
- Given identical initialization data, model produces identical initial ActNorm scale/shift
- `num_train_samples` is set to actual training set size (e.g., ~1.19M for 99% of 1.2M)

---

### Phase 3: Fix Noise Application Parity

**Goal**: Produce identical noisy inputs given identical seeds.

**Changes to `src/lightning_data.py` and `src/lightning_module.py`**:

1. Add `SeededNoiseGenerator` class:
   - Accepts seed and width_noise
   - Produces noise via `torch.rand_like(x) * width_noise`
   - When `fixed_noise=True`: generates noise once at dataset creation and reuses
   - When `fixed_noise=False`: generates fresh noise per epoch using epoch seed

2. Match old `MyDataLoader` noise logic:
   - Use `torch.randperm` for index shuffling (same as line 66 of old code)
   - Apply noise after indexing, before returning batch
   - Use `torch.clone` to prevent in-place modifications (same as line 81)

3. Ensure `width_noise` default matches legacy (5.0e-6)

**Verification**:
- With `seed=2026`, `width_noise=5e-6`, `fixed_noise=True`: first 10 batches from new code match old code exactly

---

### Phase 4: Fix Scheduler and Optimizer Parity

**Goal**: Support all legacy scheduler types in Lightning module.

**Changes to `src/lightning_module.py`**:

1. Extend `configure_optimizers()` to support:
   - `step`: `torch.optim.lr_scheduler.StepLR`
   - `reduce_on_plateau`: `torch.optim.lr_scheduler.ReduceLROnPlateau`
   - `one_cycle_lr`: `torch.optim.lr_scheduler.OneCycleLR`
   - `cycle_lr`: `torch.optim.lr_scheduler.CyclicLR`
   - `multi_step_lr`: `torch.optim.lr_scheduler.MultiStepLR`

2. Compute `steps_per_epoch` identically to legacy:
   ```python
   steps_per_epoch = num_train_samples // batch_size  # (accounting for drop_last)
   ```

3. Ensure LR is identical per step by matching scheduler configuration exactly

**Verification**:
- LR values match old code for first 100 steps with identical seed

---

### Phase 5: Fix LightningRunner to Use Current API

**Goal**: Ensure `src/main.py --lightning` works with current implementation.

**Changes to `src/lightning_runner.py`**:

1. Update `LightningRunner.__init__` to use current `CaloINNDataModule` signature:
   - Old: `CaloINNDataModule(params)` (dict)
   - New: `CaloINNDataModule(data_path=..., batch_size=..., ...)` (individual args)

2. Update `CaloINNLightningModule` construction to use current signature

3. Ensure `train()`, `generate()`, `save()`, `load()` work with fixed implementation

**Verification**:
- `python src/main.py --lightning params/pions_odd_discrete.yaml` starts training without error

---

### Phase 6: Update Dependencies (torch 2.x, lightning 2.6.1)

**Goal**: Enable upgrade to modern PyTorch and Lightning versions.

**Files to modify**: `pyproject.toml`, `requirements.txt`

**Risks**:
- `FrEIA` library may not be compatible with torch 2.x (written for torch 1.x)
- May need to pin FrEIA to a specific commit or fork it

**Implementation**:
1. First, test current code with torch 2.x (may work due to backward compatibility)
2. If FrEIA fails, examine FrEIA source to identify torch 1.x specific APIs
3. Create local patched version of FrEIA if needed
4. Update `pyproject.toml`:
   ```toml
   torch>=2.0
   lightning==2.6.1  # or compatible version
   freia @ git+https://github.com/...#commit-with-torch2-support
   ```

**Verification**:
- `uv run python -c "import torch; import lightning; print(torch.__version__, lightning.__version__)"` works
- Full training loop runs for 10 steps without errors

---

### Phase 7: End-to-End Parity Test

**Goal**: Confirm bit-for-bit parity between old and new code.

**Test configuration**:
- Config: `CaloINN/params/pions_odd_discrete.yaml`
- Seed: 2026
- Steps: 10 (to keep test fast)
- Compare:
  - Loss values per step (must match to 1e-6)
  - Model weights after 10 steps (must match to 1e-6)
  - LR schedule (must match exactly)
  - Noised input tensors (must match exactly)

**Verification**:
```
Old code losses: [32382.84, 31914.71, ...]
New code losses: [32382.84, 31914.71, ...]
max |diff| < 1e-6 ✓
```

---

## Implementation Order

```
Phase 1 (streaming dataset)
    ↓
Phase 2 (init parity)
    ↓
Phase 3 (noise parity)
    ↓
Phase 4 (scheduler parity)
    ↓
Phase 5 (LightningRunner fix)
    ↓
Phase 6 (dependency update) ← may need FrEIA patching
    ↓
Phase 7 (parity test)
```

---

## File Inventory

| File | Status | Changes |
|------|--------|---------|
| `src/streaming_data.py` | **NEW** | HDF5IterableDataset for memory-efficient streaming |
| `src/lightning_data.py` | Modify | Replace TensorDataset; fix num_train_samples; integrate streaming |
| `src/lightning_module.py` | Modify | Fix num_train_samples; add all schedulers; fix init; add seeded noise |
| `src/lightning_runner.py` | Modify | Update to current API signatures |
| `src/main.py` | No change | Should work after phases complete |
| `src/trainer.py` | No change | Legacy reference, kept for parity comparison |
| `src/myDataLoader.py` | No change | Legacy reference |
| `src/model.py` | No change | Core CINN model |
| `pyproject.toml` | Modify | Update torch/lightning versions |
| `params/pions_odd_discrete.yaml` | No change | Reference config |

---

## Summary of Critical Fixes

1. **Memory**: Replace `TensorDataset` with `HDF5IterableDataset` → prevents RAM overflow
2. **KL scaling**: Set `num_train_samples` to actual training set size → correct loss scale
3. **Noise**: Implement seeded noise matching old `MyDataLoader` → reproducible results
4. **Schedulers**: Support all legacy scheduler types → matching LR schedule
5. **API**: Fix `LightningRunner` → `src/main.py --lightning` works
6. **Deps**: Update torch/lightning → modern stack compatibility
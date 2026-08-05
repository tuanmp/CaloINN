# Experiments

> Maintained by the ExperimentTracker agent. Source of truth: `registry.json`. Updated on demand — regenerate the tables here from the registry whenever the registry changes.

## Studies

| Study | Status | Runs | Question |
|-------|--------|------|----------|
| [hmc-test](studies/hmc-test.md) | active | 1 | Does HMC sampling improve density-ratio correction / best AUC? |

## Runs (newest first)

| Submitted | Slug | Cluster | Job | Status | Config | Study |
|-----------|------|---------|-----|--------|--------|-------|
| 2026-08-05 | [hmc_test](runs/hmc_test.md) | perlmutter | — | done | run_hmc_test.json | [hmc-test](studies/hmc-test.md) |
| 2026-08-04 | [56355982_pm_submit_batch_40GB](runs/56355982_pm_submit_batch_40GB.md) | perlmutter | 56355982 | running | — | — |
| 2026-08-04 | [56355976_pm_submit_batch_40GB](runs/56355976_pm_submit_batch_40GB.md) | perlmutter | 56355976 | running | — | — |
| 2026-08-04 | [56355970_pm_submit_batch_40GB](runs/56355970_pm_submit_batch_40GB.md) | perlmutter | 56355970 | running | — | — |
| 2026-08-04 | [56318023_pm_submit_preempt_1GPU](runs/56318023_pm_submit_preempt_1GPU.md) | perlmutter | 56318023 | killed | — | — |
| 2026-04-01 | [20260401_130938_full_training_det_u10_rew85_1e6](runs/20260401_130938_full_training_det_u10_rew85_1e6.md) | perlmutter | — | done | params.yaml | — |
| 2026-04-01 | [20260401_115204_full_training_det_u10_rew85_1e6](runs/20260401_115204_full_training_det_u10_rew85_1e6.md) | perlmutter | — | done | params.yaml | — |
| 2026-04-01 | [20260401_080529_full_training_det_u10_rew85_1e6](runs/20260401_080529_full_training_det_u10_rew85_1e6.md) | perlmutter | — | done | params.yaml | — |

## Legend

- **Status:** `submitted`, `running`, `done`, `failed`, `killed`
- **git_commit:** `unknown` for historical runs captured before tracking existed.

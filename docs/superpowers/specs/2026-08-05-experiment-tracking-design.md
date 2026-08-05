# Design: Experiment Tracking Agent

**Date:** 2026-08-05
**Status:** Approved
**Author:** pmtuan + opencode

## Problem

Experiments span multiple SLURM clusters with artifacts scattered across
`slurm_logs/`, `results/`, and `lightning_logs/`. Run metadata (job id, cluster,
status, config, git commit) is not recorded systematically, and study-level
narratives only exist as ad-hoc `_latest.txt` / `_summary.md` files. The user
cannot easily answer "what did I run, what status is it in, and what did I
learn?"

## Goals

1. **Run index** — a searchable ledger of every run: SLURM job id, cluster,
   status, config, git commit, output paths.
2. **Study narratives** — themed write-ups (question, runs, findings, next
   steps) that group related runs.
3. **Offline + online access** — markdown committed to a git branch is the
   offline source of truth and is browsable on github.com from anywhere.

## Non-goals

- No automatic job-submission hooks or cron-based polling.
- No Notion/API sync (GitHub markdown chosen over Notion).
- No deterministic parser scripts for the index (pure-LLM reconcile; YAGNI).
- The `ExperimentTracker` agent never launches training or touches HDF5 data.

## Terminology

- **Run** — one execution (a SLURM job / training run). Lives in
  `results/YYYYMMDD_HHMMSS_<name>/`, has a job id.
- **Study** — a themed investigation (e.g., "HMC test", "noise restructure",
  "MCMC density-ratio") that groups multiple runs under a narrative.

## Storage layout

```
docs/experiments/
├── INDEX.md            # human-readable: tables of studies + runs, links
├── registry.json       # machine-readable source of truth (agent updates this)
├── studies/
│   └── <study-slug>.md # narrative: question, runs, findings, next steps
└── runs/
    └── <run-slug>.md   # per-run detail: config, job id, cluster, metrics, artifacts
```

`registry.json` is the master record. `INDEX.md`, per-run pages, and study pages
are views maintained by the agent. The whole `docs/experiments/` tree is
committed to a dedicated `experiment-docs` branch and pushed to the existing
GitHub remote (`origin`, `https://github.com/tuanmp/CaloINN.git`). The
`remote` URL (which embeds a plaintext token) must NOT be used for pushing.

## registry.json schema

Run entry:

```json
{
  "run_id": "20260401_130938_full_training_det_u10_rew85_1e6",
  "slug": "20260401_130938_full_training_det_u10_rew85_1e6",
  "submitted_at": "2026-04-01T13:09:38",
  "cluster": "perlmutter",
  "job_id": "56318031",
  "status": "done",
  "config": "params/pions_odd.yaml",
  "config_hash": "sha1...",
  "git_commit": "a219826",
  "script": "batch/pm_submit_preempt_1GPU.sh",
  "results_dir": "results/20260401_130938_full_training_det_u10_rew85_1e6",
  "slurm_out": "slurm_logs/pm-slurm-56318031-pm_submit_preempt_1GPU.sh.out",
  "slurm_err": "slurm_logs/pm-slurm-56318031-pm_submit_preempt_1GPU.sh.err",
  "final_metrics": { "val_loss": 7574.05 },
  "artifacts": [],
  "study_id": null,
  "notes": ""
}
```

Study entry:

```json
{
  "study_id": "hmc-test",
  "title": "HMC test",
  "status": "active",
  "question": "Does HMC sampling improve density-ratio correction?",
  "runs": ["..."],
  "findings": "",
  "next_steps": ""
}
```

Statuses: `submitted`, `running`, `done`, `failed`, `killed`.

## Components

### 1. Subagent — `ExperimentTracker`

- File: `.opencode/agent/subagents/experiment/experiment-tracker.md`
- `mode: subagent`, `temperature: 0.1`
- Description: "Tracks, documents, and queries ML experiments: reconciles
  SLURM logs and result dirs, maintains docs/experiments registry, writes
  study narratives."
- Permissions:
  - read/glob/grep/list: allow
  - bash: allow for `ls/find/rg/squeue/git`; git scoped to `experiment-docs`
    branch operations; push denied on the token-embedded `remote`
  - edit: allow `docs/experiments/**`; deny elsewhere (protect src/, params/)
- Responsibilities:
  - **Reconcile** — scan `slurm_logs/`, `results/`, `lightning_logs/`,
    `params/`, and `squeue -u pmtuan` to discover runs and flip stale statuses
    (running -> failed/done).
  - **Log-as-you-work** — after a run is submitted or finishes, capture config
    path, job id, cluster, git commit, script.
  - **Query** — answer questions by reading `docs/experiments/`.
  - **Write narratives** — create/update study pages from grouped runs.
  - **Git sync** — commit `docs/experiments/**` on the `experiment-docs`
    branch and push to `origin`.

### 2. Skill — `experiment-tracking`

- File: `.opencode/skills/experiment-tracking/SKILL.md`
- Frontmatter description with trigger keywords: experiment, run, study, track,
  log, reconcile, slurm, results/.
- Body documents: terminology, file layout, registry schema, doc templates
  (INDEX.md table, study narrative, per-run page), scan procedures (parse
  `slurm_logs/pm-slurm-<jobid>-*.sh.{out,err}`, `results/YYYYMMDD_HHMMSS_<name>/`,
  `lightning_logs/version_*/metrics.csv`, `params/*.yaml`, `squeue -u pmtuan`),
  and git discipline (branch `experiment-docs` only; never the token-embedded
  remote; commit per update).

## Interaction model

- **On-demand:** "what did I run last week?", "log my latest run", "write up
  the HMC study" -> main agent delegates to `ExperimentTracker`.
- **Log-as-you-work:** after submitting/finishing a run, the user says "track
  it" and the agent captures the run metadata.
- **Auto-reconcile:** each delegation re-scans `squeue` + `slurm_logs` to flip
  stale statuses and discover new runs.

## Edge cases & gotchas

- `squeue` is only available on login nodes. Fallback: parse slurm `.err/.out`
  for status.
- `git_commit` for historical runs is unknown -> `"unknown"`. Captured going
  forward when the run is logged as-you-work.
- The agent must never run training, never touch HDF5, and never read/modify
  files outside `docs/experiments/` plus read-only scan dirs.
- ⚠️ `.git/config` contains a plaintext GitHub token in the `remote` URL. The
  agent must not push to that URL. Token cleanup is a separate follow-up task.

## Verification

- Dry-run reconcile: after implementation, delegate to `ExperimentTracker` to
  generate the first `registry.json` + `INDEX.md` WITHOUT committing, then the
  user reviews.
- User must restart opencode for the new agent and skill to register.

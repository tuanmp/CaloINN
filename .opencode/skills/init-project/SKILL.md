---
name: init-project
description: Initialize a new research project with standard directory structure, CLAUDE.md, pyproject.toml, and skills. Use at the start of any new notebook-based research project. Run from within the project folder — the folder name becomes the project name.
argument-hint: [short-description]
disable-model-invocation: true
---

# Initialize Research Project

Set up a new research project in the current working directory.

- **Project name**: `basename $(pwd)` (e.g., `my-cool-project`)
- **Package name**: hyphens replaced by underscores (e.g., `my_cool_project`)
- **Description**: `$ARGUMENTS` if provided, else `"Research project"`
- **Assets**: `assets/` (relative to this skill's directory)
- **Cluster info**: `../slurm-info-summary/references/slurm-cluster-summary.md` (relative to this skill's directory)

## Steps

### 1. Derive names

```bash
PROJECT_NAME=$(basename $(pwd))
PACKAGE_NAME=$(echo "$PROJECT_NAME" | tr '-' '_')
ASSETS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/assets"
```

### 2. Create directory structure

```bash
mkdir -p notebooks "src/${PACKAGE_NAME}" attachements research_notes .claude/skills/report
touch "src/${PACKAGE_NAME}/__init__.py"
```

### 3. Copy static files (no substitution needed)

```bash
cp "${ASSETS}/report-SKILL.md" .claude/skills/report/SKILL.md
cp "${ASSETS}/plotting.py" "src/${PACKAGE_NAME}/plotting.py"
```

### 4. Copy and substitute CLAUDE.md

```bash
cp "${ASSETS}/CLAUDE.md.template" CLAUDE.md
sed -i "s|{{PROJECT_NAME}}|${PROJECT_NAME}|g; s|{{PACKAGE_NAME}}|${PACKAGE_NAME}|g" CLAUDE.md
```

### 5. Copy and substitute pyproject.toml

Use the skill argument (`$ARGUMENTS`) as description. If empty, use `"Research project"`.

```bash
DESCRIPTION="<$ARGUMENTS or 'Research project'>"
cp "${ASSETS}/pyproject.toml.template" pyproject.toml
sed -i "s|{{PROJECT_NAME}}|${PROJECT_NAME}|g; s|{{PACKAGE_NAME}}|${PACKAGE_NAME}|g; s|{{DESCRIPTION}}|${DESCRIPTION}|g" pyproject.toml
```

### 6. Copy and substitute .mcp.json from template + cluster info

1. **Read cluster summary** at `../slurm-info-summary/references/slurm-cluster-summary.md` (relative to this skill's directory).
   - If it does **not** exist, tell the user: *"Run `/slurm-info-summary` first to gather cluster info."* and **stop**.

2. **Extract values** for a single-GPU shared notebook job from the summary:

   | Variable | What to look for | Example (Raven) |
   |----------|-----------------|------------------|
   | `PARTITION` | Shared GPU partition (1 node, shared access, has GPUs) | `gpu1` |
   | `GPU_TYPE` | GPU model, lowercase | `a100` |
   | `CPUS` | Physical cores per GPU (total physical cores / GPUs per node) | `18` |
   | `MEM_MB` | Memory per GPU in MB (convert from GB if needed) | `125000` |
   | `TIME` | Max walltime for that partition | `1-00:00:00` |

3. **Detect CUDA module**:
   ```bash
   CUDA_MODULE=$(module avail cuda 2>&1 | grep -oP 'cuda/[\d.]+' | sort -V | tail -1)
   ```
   - If found: use the module name (e.g., `cuda/12.6`)
   - If empty: use empty string `""`

4. **Copy template and substitute**:
   ```bash
   cp "${ASSETS}/.mcp.json.template" .mcp.json
   sed -i \
     -e "s|{{PARTITION}}|${PARTITION}|g" \
     -e "s|{{GPU_GRES}}|gpu:${GPU_TYPE}:1|g" \
     -e "s|{{CPUS_PER_TASK}}|${CPUS}|g" \
     -e "s|{{MEM_MB}}|${MEM_MB}|g" \
     -e "s|{{TIME}}|${TIME}|g" \
     -e "s|{{CUDA_MODULE}}|${CUDA_MODULE}|g" \
     .mcp.json
   ```

### 7. Print summary

Show what was created and next steps:
- `uv tool install git+https://github.com/kdkyum/jlab-mcp.git` (one-time, if not already installed)
- `uv sync` to install project dependencies
- `jlab-mcp start` in a separate terminal to launch JupyterLab on a GPU node
- Start Claude Code in the project directory

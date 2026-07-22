---
name: report
description: Generate a research report for a completed notebook experiment. Use after running a notebook to create a structured markdown report with figures, tables, and conclusions in the research_notes/ directory.
argument-hint: <notebook-name-without-extension> [optional-title-override]
allowed-tools: Read, Glob, Grep, Write, Bash(mkdir:*), Bash(ls:*), Bash(date:*), Bash(printf:*)
---

# Research Report Generator

Generate a research report for the notebook `$0`.

## Gather Context

1. **Find the notebook**: Look for `notebooks/$0.ipynb`. Read it directly using the `Read` tool (the notebook is on the shared filesystem as a `.ipynb` JSON file). Extract:
   - The experiment title from the first markdown cell
   - Method description from markdown cells
   - Code cell outputs (tables, printed results, key metrics)
   - Any conclusions or summary cells

2. **Find figures**: Search for saved figures associated with this notebook:
   ```
   attachements/$0_*.png
   attachements/$0_*.pdf
   ```
   Collect all figure paths and infer descriptions from filenames.

3. **Identify key results**: From code cell outputs, extract:
   - Accuracy tables or metric summaries
   - Model comparison numbers (wd0.5 vs wd3.0)
   - Any printed summary statistics

## Write the Report

Create the report at:
```
research_notes/YYYY-MM-DD-HHMMSS_<title>.md
```

where `YYYY-MM-DD-HHMMSS` is the current date and time (e.g., `2026-02-03-143025`) and `<title>` is derived from the notebook's H1 heading (lowercased, spaces replaced with underscores, stripped of special characters). If the user provided `$1`, use that as the title instead. Use `date '+%Y-%m-%d-%H%M%S'` to get the timestamp.

Ensure `research_notes/` directory exists (create if needed).

### Report Structure

Use this template:

```markdown
# <Experiment Title>

**Date**: YYYY-MM-DD-HHMMSS
**Notebook**: `notebooks/$0.ipynb`
**Models**: <model descriptions from notebook context>

## Motivation

<Why this experiment was run. Summarize from the notebook's introductory markdown cells.>

## Method

<Concise description of the experimental procedure. Reference specific techniques (RESCAL, activation patching, etc.) and parameters used.>

## Results

<For each major result, include:>

### <Result Section Title>

<Description of what was measured.>

<If there is a table of numbers, reproduce it as a markdown table.>

<If there is a figure, embed it:>
![Figure description](../attachements/$0_NN_description.png)

<Interpretation of the result.>

## Conclusions

<Numbered list of key takeaways. Be specific about what was confirmed/refuted and what it means for the broader research question (reversal curse mechanism).>
```

### Guidelines

- **Be concise**: Each section should be 2-5 sentences unless the results warrant more detail.
- **Include all figures**: Every figure in `attachements/$0_*` should appear in the report.
- **Reproduce key tables**: Any printed accuracy/metric tables from cell outputs should be markdown tables in the report.
- **Note any bug fixes or methodological changes**: If cells were modified during the session, mention what was fixed and why.
- **Relative figure paths**: Use `../attachements/` for figure links since the report lives in `research_notes/`.
- **No speculation**: Only report what the data shows. Flag ambiguous results as such.

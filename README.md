# NLP1 Project: Bias in Bios

Course project for NLP1 (MSc AI, University of Amsterdam) by Isha, Diya and Ioana.

We predict a person's occupation (28 classes) from their online bio
([`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios), De-Arteaga et al., 2019),
compare 3 representations × 3 architectures with cross-validation and Bayesian estimation, and measure
whether the classifiers work equally well for women and men.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # also installs src/ in editable mode, so `import bib` and `import part1` work
```

The dataset is downloaded from Hugging Face on first use. A GPU is used when available (CUDA); otherwise
everything runs on the CPU (about 40 minutes for the full notebook on an Apple M4).

## Run

- **Main notebook:** `notebooks/bias-in-bios-week1.ipynb`. Run top to bottom; results are cached in
  `results/week1/` (git-ignored), so re-running loads them instead of retraining.
- **Only the 3 × 3 grid, from a terminal:** `python scripts/run_grid.py` (needs the splits, tokenizer and
  tuned settings the notebook writes first).
- **Explanatory diagrams:** `python scripts/make_diagrams.py`.

## Layout

| Path | Contents |
| --- | --- |
| `notebooks/` | the main notebook (§0 setup … §13 AI-assistance log); `scratch/` for experiments |
| `src/bib/` | our code: tokenizer, representations, models, tuning, metrics, cross-validation, scrubbing, plots |
| `src/part1/` | the course tutorial's code and notebooks (unchanged) |
| `scripts/` | grid runner and diagram generator |
| `figures/week1/` | all figures (PNG) produced by the notebook, plus `diagrams/` |
| `docs/` | `oral_prep.md` (likely oral questions), `ai_assistance_log.md` (AI use), `guides/` (PDF study guides for the oral: full and quick versions, project and tutorial), `reference/` (the paper) |
| `results/week1/` | cached tokenizer, folds, tuning and CV results (not in git) |

## AI assistance

We used Claude Code as a coding assistant. Every delegated task and what we checked is logged in
`docs/ai_assistance_log.md`.

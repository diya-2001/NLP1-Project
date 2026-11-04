# CLAUDE.md — NLP1 Project 1: Bias in Bios

UvA MSc AI, NLP1 course project. Team: Isha (owns the main notebook), Diya, Ioana.
Repo `diya-2001/NLP1-Project`; branches `main`, `isha-working`, `diya-working`, `ioana-working`.

## How to work here
- **Teach as you go.** Every choice must be defended in an oral exam. Explain the why in plain
  language and define jargon on first use (macro-F1, stratification, TPR, ROPE, BPE, posterior…).
- Work a **whole notebook section at a time**, then summarise.
- Refer to cells by their **tags**, never by number: code cells start with `# [4a] …`, markdown
  cells with a numbered heading `### 4.1 …`.
- Per step: markdown "what we do and why" → code → markdown "what we see / so what"
  (what we saw → why it matters → what we decided). Use **real numbers from outputs only**; if a
  cell hasn't run, run it (if quick) or ask. Placeholders look like `⟨…⟩`.
- Charts: one message per chart, takeaway as the title, annotate key points, women/men colours
  consistent throughout. Save PNGs to `figures/week1/` at dpi 200.
- Maintain `docs/oral_prep.md` (likely TA questions + short answers per section) and
  `docs/ai_assistance_log.md` (date | tool | delegated | what was checked/changed). Log your own work.
- **Never modify `src/part1/`** (tutorial code). Reusable code goes in `src/bib/` (documented modules).
- **No commits/pushes unless asked.** Ask before jobs > ~5 min; test on a small subsample first.
- Judgement calls: give options + a recommendation.
- Teammates: experiment in `notebooks/scratch/`, share code via `src/bib/` (notebooks merge badly).
- VS Code may hold **unsaved notebook edits**: check the file's mtime vs. recent outputs before
  editing `notebooks/*.ipynb`, and ask Isha to save first.

## Assignment (Project 1)
3 representations × 3 architectures, compared by cross-validation with a relevant metric and
Bayesian estimation with a motivated ROPE. Oral the week after the deadline; up to 5 backup slides
(data & metrics, features, architectures, results, extra). Same dataset in Projects 2–3 and the
final blog → reproducible code, documented sources and AI use. Brief: `src/part1/project.ipynb`.

## Environment
- `.venv/` (Python 3.12), use `.venv/bin/python`; kernel "NLP1 (.venv)". Don't use conda.
- `pip install -e .` (via `requirements.txt`) → `import part1`, `import bib` work anywhere.
- numpy 2.5, **pandas 3.1 (pyarrow strings → `.str` regex is RE2: no lookarounds/backrefs)**,
  scikit-learn 1.9 (`multi_class` arg removed), torch 2.15, tokenizers 0.23, baycomp 1.0.3.
- MacBook M4, 16 GB: **torch on CPU** (sparse ops unreliable on MPS); **always CSR sparse**, never dense doc vectors.
- Data: `load_dataset("LabHC/bias_in_bios")` (cached). HF_TOKEN warning is harmless.
- Paths: notebook `notebooks/bias-in-bios-week1.ipynb`; cached outputs `results/week1/` (git-ignored);
  figures `figures/week1/` (committed); `docs/reference/bias-in-bios.pdf`.

## Dataset facts
- Columns `hard_text` (bio minus the first, label-giving sentence), `profession` 0–27, `gender`
  (0 male / 1 female, **binary, inferred from pronouns**). Label order = alphabetical list `PROFESSIONS`.
- train 257,478 / dev 39,642 / test 99,069; 46.1% female in each; no empty texts.
- Train duplicates: 676 rows / 294 texts; **15 texts with conflicting occupation labels**, 0 with
  conflicting gender. 95 dev / 256 test bios also in train.
- Imbalance: professor 29.8%, smallest rapper 911 (84×). Baselines: majority acc 0.298 /
  macro-F1 0.016; class-share random acc 0.125 / macro-F1 0.036.
- % female ranges 9.7 (rapper) → 92.9 (dietitian). Smallest cells: female rapper 88, female dj 136.

## Decisions
- Task: 28-way occupation from `hard_text`. **Gender never a model input** — evaluation only.
- Grid `{arch}_{rep}`: reps `indicator`, `bow`, `tfidf` (ours: sublinear tf × idf, L2-norm, **idf
  fit on the training fold only**); archs `loglin` (sklearn, CSR), `mlp` (tutorial `SparseMLP`),
  `deep` (ours: 2 hidden layers ~256→128, ReLU, dropout ~0.3, sparse first layer). Fallback: L-shape.
- Metrics: **macro-F1** primary; also accuracy, balanced accuracy. Fairness: per-occupation TPR gap
  (female − male recall) and RMS over occupations, on pooled out-of-fold predictions, cells ≥ 10.
- Splits: CV only inside `train` (dev/test untouched this week); `dedup` (drop the 15 conflicting
  texts entirely, keep one copy of other duplicates) = 257,081 bios; **`POOL_SIZE=None` (full dedup)**,
  **`N_FOLDS=3`, `SEEDS=[0,1]`** = 6 scores/system (Isha found 5×3 ≈ 1.5 h too heavy; 3×2 ≈ 30 min; cost:
  Nadeau–Bengio factor 1/6+1/2 vs 1/15+1/4 → wider posteriors); `StratifiedKFold` on `Y*2+G`;
  folds in `results/week1/splits.json` (reload with `bib.data.load_pool`). Old 5-fold files in `results/week1/archive_5fold/`.
- Tuning: one knob per system (loglin `C` ∈ {0.3,1,3}; MLPs lr ∈ {1e-3,3e-3}) on an inner 90/10 split of
  seed-0/fold-0 training data → `results/week1/tuned_hparams.json`. Reason: TF-IDF rows have norm 1 vs ~9
  for counts, so a shared C/lr could be unfair. (5-fold run with C ∈ {1,10,100} picked C=1 everywhere at
  full scale; the 20k pilot had favoured C=10 for TF-IDF — data size changes the optimum.)
- Notebook style: every code cell is preceded by a `> **This cell:** …` plain-language note; plotting code
  lives in `bib/plots.py` so cells only compute + call one plot. Isha wants basic language, no jargon
  without explanation.
- Notebook is complete §0–§13 (§9 Bayesian ROPE 0.01 + sensitivity 0.005/0.02; §10 fairness; §11 pronoun
  scrubbing of the best system + log-linear she/he weights).
- Neural training: AdamW, batch 128, ≤ 20 epochs, early stopping on **inner hold-out macro-F1**, patience 3.
- Grid can run from a terminal: `python scripts/run_grid.py` (cached per system in `results/week1/*.pkl`).
- Notebook builder lived only in a session scratchpad; edit the notebook directly from now on.
- Tokenizer: own BPE on all `dedup` train texts (unsupervised, same for all systems), `VOCAB_SIZE=10_000`,
  `MIN_FREQ=5`, NFKC+Lowercase, pre-tokeniser `Sequence([Punctuation(), Metaspace()])`.
  Saved `results/week1/bpe_lower_punct_v10000_mf5.json`. Pool texts encoded once into `IDS`.

## Tutorial code: reuse vs not
- Reuse: `analysis.BenavoliCorrelatedT` (diff = **B − A**; `.probs(rope)` → P(A), P(tie), P(B)),
  `analysis.plot_box_and_points`, `smlp.SparseMLP` / `scipy_to_torch_sparse` / `SparseDataset` /
  `sparse_collate` / `train_one_epoch` / `evaluate` (log-prob outputs + NLLLoss).
- Don't: `loglin.fit`, `smlp.train_neural_network_model` (build dense vectors; the latter also
  picks the best epoch on the validation fold → optimistic; use an inner hold-out),
  `experiment.cross_validation_split` (unstratified), `experiment.train_tokenizer` (AG News),
  `clslib.TokenIndicator.__len__` (uses a global), `analysis.make_data_frame` (accuracy only).

## References
De-Arteaga et al. 2019 (FAccT, dataset); Ravfogel et al. 2020 (ACL, HF version); Romanov et al. 2019
(NAACL, RMS TPR gap); Sennrich et al. 2016 (BPE); Benavoli et al. 2017 (JMLR, Bayesian comparison);
Corani & Benavoli 2015 (correlated t); Nadeau & Bengio 2003 (CV variance correction).

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
  consistent throughout. Save PNGs to `figures/partN/` (the current project part) at dpi 200.
- Maintain `docs/oral_prep.md` (likely TA questions + short answers per section) and
  `docs/ai_assistance_log.md` (date | tool | delegated | what was checked/changed). Log your own work.
- **Never modify `src/part1/`** (tutorial code). Reusable code goes in `src/bib/` (documented modules).
- **No commits/pushes unless asked.** Ask before jobs > ~5 min; test on a small subsample first.
- Judgement calls: give options + a recommendation.
- Teammates: experiment in `notebooks/scratch/`, share code via `src/bib/` (notebooks merge badly).
- VS Code may hold **unsaved notebook edits**: check the file's mtime vs. recent outputs before
  editing `notebooks/*.ipynb`, and ask Isha to save first.

## Oral prep material (made 2026-11-06)
- Study guides (Claude Docs): Project 1 Study Guide https://claude.ai/artifact/L1Mjd8RSgDbZfS2zKBWWNX ·
  Tutorial 1 Concepts and Questions https://claude.ai/artifact/JkNEv1RtAf3KgYLddXbVhx
- Quick (skim) versions: Project 1 Quick Guide https://claude.ai/artifact/2u2cxpQxVJTjvXL6mJrDLa ·
  Tutorial 1 Quick Guide https://claude.ai/artifact/581RA1YiponF6p9WbCGr5R
- PDF snapshots of all four guides (exported 2026-11-09): `docs/guides/*.pdf` — re-export if the docs change.
- Oral slide deck (7 slides, Slides type): https://claude.ai/artifact/7nq9J7hr8YrYf25BkeQ1XC
- Oral format: 15 min per group, 2 questions per student (one per "Get to work!" item + a follow-up,
  can be about the tutorial); slides = overview + one per Get-to-work item 2–6 (+1 extra), max 7, PDF by
  Friday 23:59, mandatory, not graded, no written-out answers.

## Project 2 (started 2026-11-17): embeddings → compositional encoders
- Brief: `src/part2/project.ipynb` (6 Get-to-work items: choose embeddings + coverage; what they capture;
  embed-and-pool; LSTM/GRU; search + variance; error analysis). Tutorial: `src/part2/tutorial.ipynb`.
  Never modify `src/part2/` (tutorial code), same rule as part1.
- **Naming by project part, not course week:** `notebooks/bias-in-bios-part1.ipynb` (renamed from week1 on
  2026-11-17), `notebooks/bias-in-bios-part2.ipynb`; `results/partN/`, `figures/partN/`.
  Pretrained embeddings go in `data/embeddings/` (git-ignored).
- **Heavy runs (LSTM grids) go to Snellius**: training lives in scripts that run there; the notebook loads
  cached results. Local Mac only for exploration, coverage analysis and small timing tests.
- Keep part 2 comparable with part 1: same pool (257,081 bios), same 6 splits (`results/part1/splits.json`),
  same metrics (macro-F1 + TPR gaps), same Bayesian ROPE ±0.01; week-1 best = mlp_tfidf 0.754.
- Plan (agreed 2026-11-17): 0 setup → 1 choose embedding model (GloVe vs fastText vs Common-Crawl vectors;
  coverage, tokenizer consistency) → 2 what embeddings capture (neighbours of job words; he−she direction vs
  % women) → 3 embed-and-pool (pooling, frozen/fine-tuned, dims) → 4 LSTM/GRU (frozen/trainable/random,
  hidden, bidirectional, word dropout, data-size curve) → 5 staged search + variance → 6 error analysis on
  the untouched `dev` split (part2 BDA model or sklearn L1) → 7 fairness extra → 8 oral prep.
- Step 1–2 done (2026-11-17): **chosen embeddings = fastText Common Crawl 300-d** (`data/embeddings/crawl-300d-2M.vec.zip`);
  tokenizer = lowercase + plain quotes + NLTK + per-model back-off (`bib.embeddings.tokenize/backoff`); vectors
  restricted to our vocabulary cached in `results/part2/emb_*.npz`, tokens in `results/part2/tokens_nltk.pkl`.
  Key numbers: coverage 99.32%; quick test (mean-pool + loglin, seed0/fold0) 0.733 vs wiki 0.721, GloVe-50 0.562;
  he–she axis vs % women r = 0.90. GloVe 50-d vs 300-d confounds size and corpus → test size in §4.
- Don't reuse as-is: `part2.experiment.train`/`cross_validation` (best epoch chosen on the validation fold,
  accuracy only, re-tokenizes per fold, mutates kwargs dicts). Reuse `part2.clslib` encoders/regressor,
  `PretrainedEmbeddings` (but restrict rows to the corpus vocabulary), `part2.analysis` BDA model.
- Part 2 training code (2026-11-18): `bib/neural.py` (table rows 0=<pad>, 1=<unk>=mean vector; `pool_features`
  precomputes frozen pooled vectors; `DenseModel`/`PoolModel`/`RNNModel` (LSTM/GRU, bidirectional, word dropout as
  the tutorial's `LSTMEncoder`); `fit` = Part 1 recipe + grad clip 5; trainable tables use sparse grads + SparseAdam on
  CPU/CUDA, dense on MPS) and `bib/runs.py` (`system(kind, **changes)` → auto name like `pool-mean-mlp256-finetune`,
  `rnn-gru-h128-bi-frozen`; `groups()`; `run()` caches `results/part2/{name}.pkl`, Part-1-compatible; `Data.from_disk`
  needs only `labels.npz`, `ids_*.npz`, `emb_*.npz`, `splits.json` → Snellius-ready). CLI: `scripts/run_part2.py`
  (logs in `results/part2/logs/`). Diagrams: `scripts/make_diagrams_part2.py` → `figures/part2/diagrams/`.
- Device (timed 2026-11-18, 20k bios): LSTM 6.1 s/epoch MPS vs 13.1 CPU; bi-GRU 5.5 vs 19.0; pooled MLP faster on CPU.
  `neural.device_for`: CUDA > MPS for rnn > CPU. Full LSTM run ≈ 1.5–2 h per system on MPS → LSTM grid on Snellius.
- §4 search is staged: stage 1 = pooling {mean,max,meanmax} × head {loglin, mlp256, mlp256x256}, frozen; stage 2 =
  winner × {finetune, random, PCA 50/100, GloVe-50}; then Bayesian vs Part 1 mlp_tfidf/loglin_tfidf.
- §4 done (2026-11-18, notebook run end to end, 71 cells): stage 1 winner `pool-mean-mlp256` 0.746 (mean ≫ max 0.659;
  1 hidden layer +0.018 over loglin; 2nd layer and meanmax = practically equal). Stage 2: **best `pool-mean-mlp256-finetune`
  0.762** (best epoch ≈ 2, overfits fast); random start 0.731; PCA 50/100 → 0.668/0.723; GloVe-50 0.586. vs Part 1
  mlp_tfidf 0.754: +0.008, P(equal) 0.86 (practically equal); vs loglin_tfidf 0.749: P(better) 0.97. Stage 1+2 took 16 min.
- §5 (2026-11-18): `runs.group(name, best)` → pool_stage1/2, pool_datasize, rnn_stage1 (tutorial-like LSTM h128 uni
  last-state frozen + head mlp256, and 7 one-change variants: gru, bi, h256, readout mean, finetune, random, wd0.25),
  rnn_stage2 (combined winner × embeddings × data 10/30%). `RNNModel(readout="last"|"mean")`. `run_part2.py --quick`
  = 2 epochs on 10k bios + time estimate. **RNN grid runs on a teammate's NVIDIA GPU** (handover doc:
  https://claude.ai/artifact/JCTKzZVithnoatHGRj96zQ); data via `scripts/make_gpu_bundle.py` →
  `results/part2_gpu_bundle.zip` (205 MB: splits.json, labels.npz, emb/ids_ft_crawl.npz; tested from a clean folder).
  Results come back as `results/part2/rnn-*.pkl`. Recap doc: https://claude.ai/artifact/3dM1CXSRJ1u3xQ92T7sWAf.
- Part 2 study guide (Claude Docs, 2026-11-18, items 1–3 full, item 4 design only — add LSTM results when back):
  "Bias in Bios: Project 2 Study Guide" https://claude.ai/artifact/5oZJ3VSosEeuDTxVE91SKP. Still due once Part 2 is done:
  Tutorial 2 guide, quick versions of both, PDFs in `docs/guides/`, 7 slides.

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
- Paths: notebook `notebooks/bias-in-bios-part1.ipynb`; cached outputs `results/part1/` (git-ignored);
  figures `figures/part1/` (committed); `docs/reference/bias-in-bios.pdf`.

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
  `deep` (ours, v2 by a teammate on 2026-11-06: sparse 10k→128 + LayerNorm + ReLU, then a residual
  128→128 block with LayerNorm, ReZero alpha starting at 0, dropout 0.3, 20% input-token dropout;
  1,300,765 weights). v1 (plain 256→128 stack, 2.6M weights) scored below the MLP (0.735 vs 0.750 mean,
  worst on TF-IDF) because TF-IDF's tiny scale starved layer 2; v2 matches the MLP.
  `models.DEVICE` = CUDA if available else CPU (loglin always CPU).
- `cv.run_system` cache is reused only if folds AND model settings (`hp`) match.
- Metrics: **macro-F1** primary; also accuracy, balanced accuracy. Fairness: per-occupation TPR gap
  (female − male recall) and RMS over occupations, on pooled out-of-fold predictions, cells ≥ 10.
- Splits: CV only inside `train` (dev/test untouched this week); `dedup` (drop the 15 conflicting
  texts entirely, keep one copy of other duplicates) = 257,081 bios; **`POOL_SIZE=None` (full dedup)**,
  **`N_FOLDS=3`, `SEEDS=[0,1]`** = 6 scores/system (Isha found 5×3 ≈ 1.5 h too heavy; 3×2 ≈ 30 min; cost:
  Nadeau–Bengio factor 1/6+1/2 vs 1/15+1/4 → wider posteriors); `StratifiedKFold` on `Y*2+G`;
  folds in `results/part1/splits.json` (reload with `bib.data.load_pool`).
- Tuning: one knob per system (loglin `C` ∈ {0.3,1,3}; MLPs lr ∈ {1e-3,3e-3}) on an inner 90/10 split of
  seed-0/fold-0 training data → `results/part1/tuned_hparams.json`. Reason: TF-IDF rows have norm 1 vs ~9
  for counts, so a shared C/lr could be unfair. (5-fold run with C ∈ {1,10,100} picked C=1 everywhere at
  full scale; the 20k pilot had favoured C=10 for TF-IDF — data size changes the optimum.)
- Notebook style: every code cell is preceded by a `> **This cell:** …` plain-language note; plotting code
  lives in `bib/plots.py` so cells only compute + call one plot. Isha wants basic language, no jargon
  without explanation.
- Notebook is complete §0–§13 (§9 Bayesian ROPE 0.01 + sensitivity 0.005/0.02; §10 fairness; §11 pronoun
  scrubbing of the best system + log-linear she/he weights).
- Neural training: AdamW, batch 128, ≤ 20 epochs, early stopping on **inner hold-out macro-F1**, patience 3.
- Grid can run from a terminal: `python scripts/run_grid.py` (cached per system in `results/part1/*.pkl`).
- Notebook builder lived only in a session scratchpad; edit the notebook directly from now on.
- Tokenizer: own BPE on all `dedup` train texts (unsupervised, same for all systems), `VOCAB_SIZE=10_000`,
  `MIN_FREQ=5`, NFKC+Lowercase, pre-tokeniser `Sequence([Punctuation(), Metaspace()])`.
  Saved `results/part1/bpe_lower_punct_v10000_mf5.json`. Pool texts encoded once into `IDS`.

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

# Project 1 Notebook — Section-by-Section Walkthrough

*Companion to `project1_bias_in_bios.ipynb` · NLP1 Project 1 · Bias in Bios*

For every section of the notebook this document explains **what we do**, **what we get out of it**, **how it relates to the project requirements**, **questions the TA could ask** (with the gist of a good answer), and **what more could be done**.

---

## Contents

- [Part A — The big picture](#part-a--the-big-picture)
  - [A.1 What the notebook is for](#a1-what-the-notebook-is-for)
  - [A.2 Requirement → section map](#a2-requirement--section-map)
  - [A.3 The 3 × 3 design explained](#a3-the-3--3-design-explained)
  - [A.4 Data flow through the notebook](#a4-data-flow-through-the-notebook)
- [Part B — Section by section](#part-b--section-by-section)
  - [§0 Setup](#0--setup)
  - [§1 Data & EDA](#1--data--exploratory-analysis)
  - [§2 Splits](#2--experimental-splits)
  - [§3 Tokenizer](#3--tokenizer)
  - [§4 Representations](#4--representations)
  - [§5 Architectures](#5--architectures)
  - [§6 Metrics](#6--metrics)
  - [§7 Cross-validation runner](#7--cross-validation-runner)
  - [§8 Results](#8--results)
  - [§9 Bayesian comparison](#9--bayesian-comparison)
  - [§10 Fairness analysis](#10--fairness-analysis)
  - [§11 Extra](#11--extra-explicit-gender-indicators)
  - [§12 Slide summary](#12--summary-for-the-backup-slides)
  - [§13 AI-assistance log](#13--ai-assistance--sources-log)
- [Part C — Cross-cutting questions](#part-c--cross-cutting-questions)

---

# Part A — The big picture

## A.1 What the notebook is for

The notebook answers one experimental question with a fairness twist:

> **Which combination of text representation and classifier architecture best predicts a person's occupation from their biography — and how differently does each one treat men and women in the same occupation?**

It is built so that every number on the slides traces back to a cell, every choice you might be asked about sits in one config cell, and all experiments are cached on disk so the notebook can be re-run without retraining.

## A.2 Requirement → section map

| Requirement (from `project.ipynb`) | Where it is satisfied |
|---|---|
| Choose a dataset; understand its statistics (e.g. instances per label) | §1 Data & EDA |
| Decide on segmentation: reuse / train a tokenizer, hyperparameters, training data | §3 Tokenizer |
| Represent documents as features: token indicators, bag-of-words, **+1 of your choice** | §4 Representations (R1, R2, R3 = TF-IDF) |
| Architectures: log-linear, single-hidden-layer MLP, **+1 of your choice** | §5 Architectures (A1, A2, A3 = deeper MLP) |
| Compare systems with **cross-validation** and a **relevant metric** | §2 Splits, §6 Metrics, §7 Runner, §8 Results |
| "Was the evaluation method adequate? Other aspects of performance?" | §6 (macro-F1, balanced accuracy, fairness gaps), §10 |
| **Bayesian estimation with a ROPE** you motivate | §9 Bayesian comparison |
| "Analyse the models and decide which is best" | §8 + §9 |
| Analyse errors | §8 confusions, §10 gendered confusions |
| Up to 5 backup slides (data & metrics, features, architectures, results, extra) | §12 Slide summary |
| Organised, reproducible code | §0 config, §2 saved folds, §3 saved tokenizer, §7 cached results |
| Record AI assistance and sources | §13 log |

## A.3 The 3 × 3 design explained

### What the two axes are

A text classifier in this course is two boxes:

```
text x ──► ENCODER h(x) ──► SOFTMAX REGRESSOR ──► p(occupation | x) ──► argmax ──► prediction
          (representation)   (architecture)
```

The project asks you to vary **both boxes independently**, three options each:

**Representations (how a bio becomes a vector)**

| | Name | Coordinate for token *v* | Status |
|---|---|---|---|
| R1 | Token indicators | 1 if *v* occurs in the bio, else 0 | required (tutorial) |
| R2 | Bag-of-words | number of times *v* occurs | required (tutorial) |
| R3 | TF-IDF | log(1 + count) × idf(*v*), whole vector L2-normalised | **our choice** |

**Architectures (how a vector becomes class probabilities)**

| | Name | Function | Status |
|---|---|---|---|
| A1 | Log-linear softmax regression | softmax(**W h** + **b**) | required (tutorial) |
| A2 | MLP, 1 hidden layer | softmax(**W₂** ReLU(**W₁ h** + **b₁**) + **b₂**) | required (tutorial) |
| A3 | Deeper MLP | 2 hidden layers (256 → 128), ReLU, dropout 0.3 | **our choice** |

### The grid

Crossing them gives nine systems, named `architecture_representation` in the notebook:

| | A1 `loglin` | A2 `mlp` | A3 `deep` |
|---|---|---|---|
| **R1 `indicator`** | `loglin_indicator` | `mlp_indicator` | `deep_indicator` |
| **R2 `bow`** | `loglin_bow` | `mlp_bow` | `deep_bow` |
| **R3 `tfidf`** | `loglin_tfidf` | `mlp_tfidf` | `deep_tfidf` |

### Why a full grid rather than nine unrelated systems

- **Clean attribution.** Comparing along a **column** (same architecture, different representation) isolates the effect of the representation. Comparing along a **row** (same representation, different architecture) isolates the effect of the architecture.
- **Interactions become visible.** Maybe TF-IDF helps the log-linear model (which is sensitive to feature scale) but makes no difference to the MLP (which can learn its own weighting). You can only see that if both cells exist.
- **It is what §9 compares.** The Bayesian comparisons are exactly the within-column and within-row pairs: 3 pairs per column × 3 columns + 3 pairs per row × 3 rows = **18 comparisons**.

### The cheaper "L-shape"

If compute is tight, run 5 of the 9 cells:

1. The whole **A1 column** (three representations with log-linear — minutes each).
2. The **row of the best representation** across A2 and A3.

That still covers "three representations" and "three architectures" as the checklist asks, just not all interactions.

### Why these two "own choices"

- **TF-IDF (R3).** Raw counts grow with bio length and give the same importance to *he* (in almost every bio) as to *chiropractic* (in very few). TF-IDF fixes both: the idf factor down-weights ubiquitous tokens; L2-normalisation removes length; log(1 + count) dampens repeats. It is a principled, classic change to the representation space and easy to defend.
- **Deeper MLP (A3).** The checklist explicitly suggests "a modification of the MLP". Adding a second hidden layer and stronger dropout tests whether more non-linear capacity helps separate 28 heavily overlapping occupations, and whether more regularisation helps with ~5,000 sparse inputs.

## A.4 Data flow through the notebook

```
§1  load splits ─► EDA ─► PI_FEMALE (% female per occupation)
§2  stratified pool of train ─► TEXTS, Y, G + FOLDS (saved to folds.json)
§3  BPE tokenizer (saved) ─► IDS cache (text → token ids)
§4  REPRESENTATIONS  {indicator, bow, tfidf}   (factories, fitted per fold)
§5  ARCHITECTURES    {loglin, mlp, deep}       (fit on train fold, predict val fold)
§6  evaluate(y, ŷ, g) ─► acc, macro_f1, bal_acc, per-occupation gaps, rms_gap
§7  run_system(rep, arch) ─► RESULTS_BY[name] = list of fold records (cached .pkl)
§8  scores table, box plots, per-class report, confusions  ─► BEST
§9  fold_scores ─► BenavoliCorrelatedT ─► P(A), P(tie), P(B), effect size
§10 gaps vs PI_FEMALE, gendered confusions
§11 scrubbed rerun, swap test, weight inspection
```

---

# Part B — Section by section

## §0 · Setup

### What we do
- Load autoreload, imports (numpy, pandas, torch, scipy, sklearn, datasets, tokenizers) and the course package `part1`, using the same path trick as the tutorial (`root = Path.cwd().parent`).
- One **configuration cell** with every experimental choice: `SEEDS`, `K`, `SUBSAMPLE`, `VOCAB_SIZE`, `MIN_FREQ`, `PRIMARY`, `ROPE`, `ROPE_GAP`, `MIN_CELL`, `DEVICE`, `RERUN`, and the results folder.
- Global random seeds.

### What we get
A single place where all decisions live. Changing a decision means editing one line and re-running.

### How it relates to the brief
"Be organised … share a reproducible version of your experiments." Seeds + one config cell + saved artefacts = reproducibility.

### Possible questions
- **Why `SUBSAMPLE = 100,000` instead of the full ~257k?** Compute: the tutorial's MLP took ~20 min per 5-fold run on 120k AG News items; nine systems on 257k longer texts would take many hours on CPU. The pool is stratified by occupation × gender, so proportions are preserved. Trade-off: smaller folds → smaller rare cells → noisier macro-F1 and gap estimates.
- **Why `K = 5`?** Each validation fold has ~20k bios — enough for stable scores — while keeping training cost at 5 fits per system. More folds give more (but correlated) observations.
- **Why one seed?** Compute. More seeds (repeated CV) add the most independent information to the Bayesian analysis; it is the first thing to increase if time allows.
- **Why CPU instead of MPS on a Mac?** Sparse matrix operations are poorly supported on MPS.

### What more can be done
- Add seeds (`[0, 1, 2]`) for repeated CV.
- Save the config dict to JSON alongside the results, so each results folder documents its own settings.
- Record the package versions (`pip freeze`) for the blog's reproducibility section.

---

## §1 · Data & exploratory analysis

### What we do
**1.0 Load.** All three splits (`train`, `dev`, `test`) as DataFrames; define the 28 occupation names, the gender mapping and the tutorial's `NamedClasses` outcome space.

**1.1 Label distribution.** Horizontal bar chart of class shares; compute the **majority-class baseline** (always predict professor) in both accuracy and macro-F1.

**1.2 Gender × occupation.** Crosstab of occupation × gender, `% female` per occupation (saved as `PI_FEMALE` for §10), bar chart with a 50% line, and a list of the smallest (occupation, gender) cells.

**1.3 Text.** Character and word length statistics; five random bios printed with their labels; regex counts of how often bios contain pronouns/titles, medical-directory templates ("NPI Number", "Having more than N years"), ALL-CAPS phrases (possible label leaks like "NURSE PRACTITIONER"), e-mail addresses and phone numbers.

**1.4 Observations cell** to fill in.

### What we get
- The **imbalance** (professor ≈ 30%; six classes < 0.5%).
- The **baseline**: ≈ 0.30 accuracy but a macro-F1 near 0.02 — the single most convincing argument for macro-F1.
- The **gender skew** per occupation and which cells are tiny.
- Evidence about **noise, templates and leakage**.

### How it relates to the brief
Step 1 of the project guide: "Spend some time understanding how the dataset looks and some statistics of it (e.g. how many instances of each classification label we have?)". It feeds the **Data & metrics** slide and justifies the metric choice in §6.

### Possible questions
- **How imbalanced is the data and what does that imply?** ~85:1 between largest and smallest class; accuracy is dominated by a few classes; a trivial model gets ~30% accuracy.
- **Why compare to a majority baseline?** To anchor the numbers: "80% accuracy" means nothing without knowing that 30% is free.
- **Overall gender is ~46% female — so is the data balanced?** Only overall; inside occupations it is heavily skewed (e.g. surgeon ≈ 15% female, nurse heavily female). The overall balance hides exactly the structure that causes bias.
- **What are the templates and why do they matter?** Auto-generated directory pages; they make some classes easy for the wrong reason (format, not content) and sometimes contain the label in capitals.
- **Is the text clean?** No: contact details, scraping debris, occasional multi-person bios. We do not clean it, to stay comparable with the literature — a choice you can defend either way.

### What more can be done
- Plot length distributions per occupation (are some classes systematically longer?).
- Count how many bios contain *no* explicit gender word.
- Check near-duplicates (templated bios that differ only in name/number).
- Heaps' law on this corpus (links back to the tutorial's suggested exercise).
- Word clouds or top tokens per occupation × gender.

---

## §2 · Experimental splits

### What we do
- Build a **CV pool**: a stratified subsample of `train` (stratified on occupation × gender).
- Extract three aligned arrays: `TEXTS`, `Y` (occupation), `G` (gender).
- Create `K` folds per seed with **`StratifiedKFold` on occupation × gender**.
- Save the fold indices to `folds.json`.
- Helper `pairs(idx)` turns indices into `(text, label)` lists — the format `part1` functions expect.

### What we get
- Identical, saved folds that **every system** uses.
- Gender stays aligned with each validation example through the index arrays.
- Rare cells (e.g. male paralegals) present in every fold.

### How it relates to the brief
"Your comparisons must be based on cross-validation." It also prepares the **paired** structure the Bayesian model needs.

### Possible questions
- **Why stratify by occupation × gender, not just occupation?** The fairness metric needs both genders of each occupation in every validation fold; with random folds some small cells could be nearly empty in a fold.
- **Why must every system use the same folds?** The correlated t-test models the *difference* between two systems fold by fold; differences are only meaningful if both systems saw the same training data and were scored on the same validation data. Pairing also removes fold difficulty as a source of noise.
- **Why not use the provided `dev` split instead of CV?** A single split gives one number per system and no sense of variability; CV gives k numbers, which is what the Bayesian analysis consumes.
- **What are `dev` and `test` for then?** Untouched: `test` for final reporting in later projects; `dev` can play the "devtest" role the tutorial mentions (e.g. picking an ensemble).
- **Why did you not use `part1.experiment.cross_validation_split`?** It only keeps `(text, label)` pairs, so gender would be lost; doing the split by indices keeps gender aligned and adds stratification.

### What more can be done
- Repeated stratified CV with several seeds.
- Run the full train split for the cheap log-linear systems and compare with the subsample (does the subsample change conclusions?).
- Group-aware splitting to keep templated near-duplicates in the same fold (prevents leakage across folds).

---

## §3 · Tokenizer

### What we do
- Train a **BPE tokenizer** on the full `train` split with `VOCAB_SIZE = 5000`, `MIN_FREQ = 2` using the tutorial's `part1.experiment.train_tokenizer` (Metaspace pre-tokenisation); save it and reload it on re-runs.
- Inspect tokenisation of a bio and of selected words (pronouns, occupation-specific words like *chiropractor, paralegal, dermatology*).
- **Cache token ids** for every bio in the pool (`IDS`), using the fast batch encoder, so encoders never re-tokenise.
- Print tokens-per-bio percentiles.

### What we get
- A domain-specific tokenizer.
- Evidence about which words survive as whole units and which are fragmented.
- Whether pronouns are single tokens (they should be: `▁he`, `▁she`), which matters for §11.
- A big speed-up for every later fold.

### How it relates to the brief
Step 2 of the project guide: "Do you want to use the tokeniser we developed? Do you need to train a new one? What hyperparameters? What data?" Feeds the **Features** slide.

### Possible questions
- **Why train a new tokenizer instead of reusing the AG News one?** Domain shift: bios use medical, legal, academic vocabulary that a news tokenizer would fragment. The dataset (~18M tokens) is big enough to train its own.
- **Why vocab size 5,000 rather than the tutorial's 1,000?** At 1k many occupation-specific words split into small, less informative pieces; 5k keeps more whole words while keeping the feature space manageable (5k dimensions). Larger → more informative units but sparser features and more parameters.
- **What does `min_frequency` do?** Prevents merges of pairs seen only once (rare names, phone-number fragments) from entering the vocabulary.
- **Is training the tokenizer on all of `train` a leak into the CV folds?** BPE is unsupervised and never sees labels, so the leak is negligible; strictly, one could fit it per fold.
- **How does BPE interact with gender?** Pronouns become single tokens (explicit gender features). Names are split into sub-word pieces, some of which may correlate with gender.
- **What would character-level segmentation do?** Tiny vocabulary, but individual characters carry almost no occupation information in a bag-of-units model (tutorial question 1).
- **Would an NLTK word tokenizer be better?** Word units are more interpretable and linguistically meaningful, but the vocabulary is open (rare words, names) and needs truncation; BPE gives a closed, controllable vocabulary (tutorial question 3).

### What more can be done
- Vary `VOCAB_SIZE` (e.g. 1k / 5k / 20k) with one cheap system and report the effect — a natural "extra".
- Lowercasing / normalisation before training.
- Compare BPE vs. a word-level tokenizer under the same classifier.

---

## §4 · Representations

### What we do
A base `Encoder` class with the tutorial's interface (`len(enc) = D`, `enc(text) → vector`) that looks up cached ids, plus:

- **R1 `TokenIndicator`** — 1 if a token occurs.
- **R2 `BagOfWords`** — token counts (`np.bincount`).
- **R3 `TfIdf`** — `fit()` computes document frequencies **on the training fold only**; smoothed idf; vector = log(1 + count) × idf, then L2-normalised.

A `REPRESENTATIONS` dictionary of **factories** (each receives the training-fold texts, so TF-IDF can be fitted per fold), and a sanity-check cell printing dimension, non-zeros, sum and max for one bio.

### What we get
Three interchangeable encoders that plug into any architecture, and a quick check that each behaves as intended (indicator max = 1, BoW sum = number of tokens, TF-IDF norm = 1).

### How it relates to the brief
Checklist: "the two representation spaces in the tutorial (token indicators and bag-of-words) and an additional one of your choice." Feeds the **Features** slide.

### Possible questions
- **Difference between indicator and BoW?** Same dimensions and same non-zero pattern; BoW adds repetition counts. For short bios most word-level tokens occur once, so they are often close; with sub-word units, repeats are more common.
- **Why might counts not help?** Short documents; counts mostly encode length; log-linear models are sensitive to feature scale, so longer bios get more extreme scores.
- **Why TF-IDF?** Down-weights ubiquitous tokens (including pronouns), up-weights distinctive ones, normalises for length.
- **Why fit idf on the training fold only?** idf is estimated from data; computing it on validation text would leak information about the evaluation data into the features.
- **What is the dimensionality and does it depend on the bio?** *D* = vocabulary size (5,000) for every bio; length only changes the values, not the dimension (tutorial "named axes" questions).
- **Can these representations distinguish "man bites dog" from "dog bites man"?** No — all three are bags; order is lost. Bigrams would capture local order.
- **Where does gender enter?** Through coordinates for *he/she/his/her/Mr/Ms* and name pieces; TF-IDF gives pronouns low idf because nearly every bio contains one.

### What more can be done
- Bag of bigrams (with a frequency cut-off or hashing).
- Sublinear counts without idf, to separate the "log" effect from the "idf" effect.
- Averaged pretrained word embeddings (as in the original paper).
- A scrubbed variant as a fourth representation (currently in §11).

---

## §5 · Architectures

### What we do
All three share one signature: `(train_pairs, val_pairs, encoder, seed) → predicted labels for val_pairs`.

- **A1 `fit_predict_loglin`** — wraps `part1.loglin.fit` / `predict` (scikit-learn logistic regression on sparse input).
- **A2 `fit_predict_mlp`** — wraps `part1.smlp.train_neural_network_model` with the tutorial defaults (hidden 128, dropout 0.1, batch 64, lr 1e-4, weight decay 1e-4, 10 epochs); returns the best-epoch predictions.
- **A3 `fit_predict_deep`** — own PyTorch model `DeepMLP`:
  - first layer applied to the **sparse** input with `torch.sparse.mm` (memory-efficient),
  - hidden layers 256 → 128 with ReLU and dropout 0.3,
  - AdamW (lr 1e-3, weight decay 1e-4), batch 256, 8 epochs,
  - `CrossEntropyLoss` on logits (= log-softmax + negative log-likelihood),
  - keeps the epoch with the best validation macro-F1.
- Helpers to build sparse matrices fold by fold and convert them to torch sparse tensors.
- An `ARCHITECTURES` dictionary.

### What we get
Three classifiers with increasing capacity that can be swapped in the runner without changing any other code.

### How it relates to the brief
Checklist: "the two kinds of architectures in the tutorial (log-linear and single-hidden-layer MLP) and an additional one of your choice (such as a modification of the MLP)." Feeds the **Architectures** slide.

### Possible questions
- **What does the log-linear model compute?** One score per occupation, *s_k = w_k · h(x) + b_k*, turned into probabilities by softmax. Decision boundaries are linear in feature space.
- **What can the MLP do that the log-linear model cannot?** Model interactions between features (e.g. "surgery" without "nurse"), i.e. non-linear decision boundaries.
- **What does the first layer of the MLP compute on a BoW vector?** A weighted sum of learned per-token vectors (columns of *W₁*) — effectively a bag of learned token embeddings.
- **How many parameters?** Log-linear: 28 × 5,000 + 28 ≈ 140k. MLP-128: 5,000 × 128 + 128 + 128 × 28 + 28 ≈ 644k. Deep: 5,000 × 256 + 256 + 256 × 128 + 128 + 128 × 28 + 28 ≈ 1.32M.
- **Why log-softmax / CrossEntropyLoss instead of softmax?** Numerical stability: avoids computing log of very small probabilities.
- **Why pick the best epoch on the validation fold — isn't that optimistic?** Slightly, yes: the same data selects the epoch and measures the score. The tutorial does the same, and the protocol is identical for all neural systems, so comparisons remain fair. A cleaner alternative is an inner validation split.
- **Why are A2 and A3 trained with different optimisers/learning rates?** A2 reuses the tutorial's code as-is; A3 is our own. This is a confound worth acknowledging; the cleanest fix is to give both the same optimiser settings.
- **Why is the decision rule argmax?** The course's default "most probable class" rule; note it ignores costs of different mistakes (tutorial question 5 on decision rules).

### What more can be done
- Tune regularisation `C` for log-linear; hidden size, dropout, learning rate for the MLPs (on an inner split, not on the test folds).
- Class-weighted cross-entropy to target imbalance.
- Mean-pooled embedding bag (fastText-style), Naive Bayes, linear SVM.
- Early stopping on an inner validation split.

---

## §6 · Metrics

### What we do
`evaluate(y, ŷ, g)` returns:
- **accuracy**,
- **macro-F1** (primary metric),
- **balanced accuracy** (mean per-class recall),
- **per-occupation TPR gender gap** = TPR(female) − TPR(male), computed only if a cell has ≥ `MIN_CELL` examples,
- **RMS gap** over occupations.

### What we get
For every fold: performance metrics *and* fairness metrics from the same predictions.

### How it relates to the brief
"A relevant metric (accuracy or else)" and "Was the evaluation method adequate? Do you want to assess other aspects of performance?" Feeds the **Data & metrics** slide.

### Possible questions
- **Why macro-F1 as primary?** Every occupation counts equally; accuracy is dominated by professor/physician/attorney; the majority baseline shows the gap (≈0.30 accuracy vs ≈0.02 macro-F1).
- **Downside of macro-F1?** Noisier across folds because tiny classes swing it, so posteriors are wider.
- **Why still report accuracy?** Comparability with the literature, which reports accuracy.
- **What is the TPR gap and why TPR?** Of the people who really have occupation *y*, the share correctly recognised, compared between women and men. It corresponds to *equality of opportunity*: qualified people should be found equally often regardless of gender.
- **Why RMS rather than the mean gap?** Gaps have signs; positive and negative gaps would cancel in a mean. RMS measures magnitude and penalises large gaps more.
- **Why `MIN_CELL`?** A TPR from 3 examples is meaningless; skipping tiny cells avoids noise dominating the RMS.
- **Fairness metrics can conflict — why this one?** It is the standard for this dataset (De-Arteaga et al.; Ravfogel et al.), and recognising qualified candidates is the harm the dataset was built to study.

### What more can be done
- FPR and precision gaps; demographic parity.
- Bootstrap confidence intervals for per-occupation gaps.
- Calibration (do predicted probabilities match observed frequencies?).

---

## §7 · Cross-validation runner

### What we do
`run_system(rep, arch, texts, tag)`:
1. For each seed and fold: build the encoder (fit on the training fold), fit the architecture, predict the validation fold.
2. Evaluate with §6.
3. Store a record per fold: seed, fold, validation indices, predictions, all metrics, gaps.
4. Print a one-line progress summary.
5. Pickle the records; reload them on later runs unless `RERUN = True`.

Then a loop over all nine (representation, architecture) pairs fills `RESULTS_BY`.

### What we get
`RESULTS_BY[system]` — a list of `K × len(SEEDS)` fold records per system, with **predictions** kept (unlike the tutorial helper, which keeps only accuracies). That makes every later analysis possible without retraining.

### How it relates to the brief
The cross-validated comparison itself. Keeping predictions enables the error analysis and fairness analysis.

### Possible questions
- **Why store predictions and not just scores?** So new metrics (macro-F1, gaps, confusions) can be computed after the fact.
- **Why is the encoder rebuilt inside each fold?** TF-IDF must be fitted on that fold's training data; doing it once on the whole pool would leak validation information.
- **How long does it take?** Log-linear: minutes per system; MLPs dominate the total. The `%%time` cell reports it — a good number to have on the slide.
- **What happens if the kernel crashes?** Finished systems are on disk and reload instantly.

### What more can be done
- Cache encoded sparse matrices per fold and representation (shared across architectures).
- Run heavy systems as a script in the background and only load results in the notebook.
- Save model weights for the best system (for ensembling in later projects).

---

## §8 · Results

### What we do
- A `scores` table with one row per (system, seed, fold).
- A summary table of mean and standard deviation of accuracy, macro-F1, balanced accuracy and RMS gap, sorted by macro-F1.
- Box plots per metric across systems.
- Pick `BEST` by mean primary metric; pool its validation predictions (first seed).
- `classification_report` (precision/recall/F1 per occupation).
- The 15 most frequent confusions (row-normalised confusion matrix).

### What we get
- The overall ranking and the spread across folds.
- Which occupations are hard and which are confused with which.

### How it relates to the brief
"Evaluate your models" and "Analyse the various models … decide which is best"; error analysis. Feeds the **Results** slide.

### Possible questions
- **Which system is best and by how much?** Answer with the mean ± std *and* point to §9 for whether the difference is practically meaningful.
- **Which classes are hardest and why?** Typically small and semantically overlapping ones (paralegal vs attorney, dj vs rapper vs composer, surgeon vs physician); professor absorbs many errors because it is huge and broad.
- **Is fold-to-fold variation large compared to differences between systems?** Read off the box plots — this is exactly why §9 exists.
- **Does the best system on macro-F1 also win on accuracy?** Not necessarily; discuss if they disagree.

### What more can be done
- Look at misclassified examples by hand (are some labels arguably wrong?).
- Per-class F1 heatmap across all nine systems.
- Learning curve: performance vs pool size.
- Ensemble of the five fold models of the best system, evaluated once on `dev`.

---

## §9 · Bayesian comparison

### What we do
- `fold_scores(system, metric)` returns fold scores in a fixed (seed, fold) order — the pairing.
- `compare(a, b, metric, rope)` fits `part1.analysis.BenavoliCorrelatedT` and returns P(A better), P(practically equivalent), P(B better) and the effect size.
- Builds the **18 within-row / within-column pairs** (see A.3) and tabulates them for the primary metric.
- Posterior plots for the comparisons that go on the slide.
- The same table for the **RMS gap** with its own ROPE.

### What we get
For each comparison, a statement like: "P(TF-IDF better than BoW under log-linear) = 0.93, P(tie) = 0.06, effect size 2.1." Plus plots showing the posterior over the mean difference against the shaded ROPE.

### How it relates to the brief
Checklist: "You should make use of Bayesian estimation to help you analyse results in relation to a region of practical equivalence (ROPE), and you can motivate the ROPE size yourself." Feeds the **Results** slide.

### Possible questions
- **What is the model?** Fold-wise differences are modelled as draws from a multivariate normal with an unknown mean difference μ and scale σ; all pairs of folds share a correlation ρ. The posterior over μ is a Student-t.
- **What is ρ and why is it needed?** CV folds share most of their training data, so their scores are correlated. Ignoring that would make the posterior over-confident. Benavoli sets ρ = 1/k (test fraction).
- **What is the ROPE and why `0.01` macro-F1?** The band of differences we treat as "no practical difference". Arguments: macro-F1 is noisier than accuracy (tiny classes swing it); the tutorial's 0.005 accuracy convention is in different units; a difference below one macro-F1 point is unlikely to matter to anyone using an occupation classifier. (Replace with your own reasoning and compare against the observed fold std.)
- **What does P(tie) = 0.6 mean?** 60% of posterior belief says the systems are practically equivalent — not that they are identical.
- **Why not a t-test and p-value?** A p-value tests a point null and says nothing about the size of the difference or its practical relevance; the posterior directly expresses belief about the difference relative to what matters (the ROPE). The tutorial argues this at length.
- **What does the effect size add?** It expresses the mean difference in units of its uncertainty; two comparisons with similar mean differences can differ in how separated the systems really are.
- **How would more seeds change the result?** More observations → narrower posterior → probabilities move away from "unsure".
- **Is running 18 comparisons a multiple-comparison problem?** In the estimation view we report posteriors rather than accept/reject decisions, so there is no error rate to inflate; still, interpret each comparison in context.

### What more can be done
- Sensitivity analysis: report how conclusions change with ROPE = 0.005 / 0.01 / 0.02.
- More seeds.
- A hierarchical model across several datasets/sub-tasks (beyond scope).

---

## §10 · Fairness analysis

### What we do
- `gap_table(system)`: per-occupation TPR gap averaged over folds, next to `% female` from §1.
- `plot_gaps`: scatter of gap vs % female, every occupation labelled, with the Pearson correlation *r* in the title; plotted for the best system and the two tutorial baselines.
- For six occupations (surgeon, nurse, paralegal, software engineer, interior designer, rapper): where do women vs men of that occupation get misclassified, with counts.

### What we get
- Whether the classifier favours the gender that already dominates each occupation (positive correlation = compounding imbalance).
- How big the gaps are for the most skewed occupations.
- Concrete gendered error patterns (e.g. female surgeons → physician; male nurses → physician; male paralegals → attorney).

### How it relates to the brief
"Do you want to assess other aspects of performance?" and the dataset's stated purpose — "whether models are more/less successful at selecting the correct occupation for certain gender–occupation pairs". Feeds the **Extra** slide.

### Possible questions
- **What does a positive correlation mean?** The model recognises women better in female-dominated occupations and men better in male-dominated ones, so its true positives are even more skewed than the occupation itself.
- **Why is that harmful?** In a hiring pipeline, the under-represented gender is found less often, making the imbalance worse over repeated use (the paper's "compounding imbalance").
- **Does a more accurate model have smaller gaps?** Not necessarily — compare `rms_gap` across systems; this is an empirical finding to report.
- **How reliable are gaps for small occupations?** Not very: few examples per gender per fold; we skip cells below `MIN_CELL` and average over folds, but uncertainty remains.
- **Why use training-set `% female`?** It describes the imbalance the model learned from; the original paper uses the test split, which gives nearly identical numbers here.

### What more can be done
- Bootstrap intervals on per-occupation gaps.
- Compare gaps across all nine systems in one figure.
- FPR / precision gaps for the same occupations.

---

## §11 · Extra: explicit gender indicators

### What we do
1. **Scrubbing.** Remove *he, she, his, her, hers, him, himself, herself, Mr, Mrs, Ms* from every bio; rerun the best system on the scrubbed text with the same folds; compare original vs scrubbed with the Bayesian machinery on macro-F1 and on RMS gap.
2. **Gender-swap test.** Train one log-linear BoW model on fold 0; predict validation bios as-is and with gender words swapped (he ↔ she, his ↔ her, Mr ↔ Ms …); report the share of predictions that change.
3. **Weight inspection.** Top-weighted tokens for nurse, surgeon, paralegal, software engineer; which occupations each pronoun pushes towards most.

### What we get
- Whether removing explicit gender words costs performance and whether it reduces gaps.
- How much individual predictions depend on gender words (flip rate).
- Direct, interpretable evidence of gender words acting as features.

### How it relates to the brief
"Extra" slide; deeper dive and error analysis beyond the checklist; directly mirrors the original paper's central experiment.

### Possible questions
- **Why can't you remove first names?** In this version the first sentence with the full name is removed, but first names still appear later; without the full-name sentence there is no reliable way to identify which capitalised word is the subject's first name. A name list would be an approximation.
- **If scrubbing reduces the gap, is the model now fair?** No — the original paper shows gender is still predictable from the remaining words (proxies such as *husband, mother, women*), and gaps remain.
- **What exactly does the swap test measure?** How much the decision depends on explicit gender words — "what if this bio used other pronouns", not "what if this person were another gender" (everything else about the bio stays the same).
- **Why is "her" ambiguous?** It maps to both *his* and *him*; the swap picks *his*, which is sometimes ungrammatical. It is an approximation.
- **Why compare with the same folds?** So the Bayesian comparison stays paired.

### What more can be done
- Scrub names with a first-name list.
- Train with scrubbed text but evaluate with original text (and vice versa).
- A gender-prediction probe on scrubbed text to show remaining gender information.
- Debiasing methods (e.g. INLP) — probably for later projects.

---

## §12 · Summary for the backup slides

### What we do
A table mapping each of the five slides to notebook outputs, plus a stub for three key findings.

### What we get
A checklist for building the slides directly from the notebook.

| Slide | Content |
|---|---|
| Data & metrics | §1 class shares, % female, baseline; metric choice; ROPE |
| Features | §3 tokenizer choices and examples; §4 R1–R3 |
| Architectures | §5 table, hyperparameters, parameter counts |
| Results | §8 table and box plots; §9 posteriors with P(A)/P(tie)/P(B), effect sizes |
| Extra | §10 gap vs % female, gendered confusions; §11 scrubbing and swap test |

### Possible questions
Any slide can be the starting point of a "why" question; the questions listed under the corresponding sections apply.

### What more can be done
Write the findings as full sentences now — they become paragraphs of the final blog post.

---

## §13 · AI-assistance & sources log

### What we do
A table (date, tool, what was delegated, what you checked or changed) and a list of sources.

### How it relates to the brief
"Keep a record of the kinds of things you delegate to any form of assistant, as you will be asked to disclose this in your blog post" and "Document your sources".

### Possible questions
- **Which parts did you write yourself and which were assisted?** Answer from the log.
- **How did you verify assisted code?** Sanity checks (§4), comparing against tutorial results, reading `part1` source.

---

# Part C — Cross-cutting questions

Questions that don't belong to one section but are likely in an oral:

| Question | Gist of a good answer |
|---|---|
| **What is your best system and would you deploy it?** | Name it with its macro-F1 and the Bayesian evidence; then say no, not as-is — the fairness analysis shows gender-correlated error rates in a high-stakes setting. |
| **What surprised you?** | e.g. counts vs indicators made little difference; the MLP did/didn't beat log-linear; gaps survived scrubbing. |
| **Biggest weakness of your setup?** | Subsample + one seed; epoch selection on validation folds; binary inferred gender; noisy small cells; A2 and A3 trained with different optimisers. |
| **Why is this a good dataset for this course?** | Large, real, clean multi-class task with a built-in second axis (gender) that makes "other aspects of performance" concrete. |
| **How will this carry into Projects 2 and 3?** | Same data and splits; later tutorials replace the fixed encoder with trainable ones, so the comparison baseline already exists. |
| **What would you change with more time/compute?** | Full training set, more seeds, tuned hyperparameters on an inner split, bigram features, scrubbing names, debiasing. |
| **How does the tokenizer affect the classifier?** | Vocabulary size controls how informative and how sparse the features are; domain matters; it also controls whether gender cues are single features. |
| **Why does the bag-of-units assumption matter here?** | Order is lost ("surgery assistant" vs "assistant surgery"), but occupation is mostly signalled by vocabulary, so bags work well; negation and roles ("worked with surgeons") are lost. |
| **Why not just remove gender from the data and be done?** | Proxies remain; occupations differ in specialisation patterns that correlate with gender; scrubbing reduces but does not remove gaps. |

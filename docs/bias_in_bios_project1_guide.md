# Bias in Bios × NLP1 Project 1 — A Complete Guide

*Dataset: [`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios) · Course: NLP1 (UvA), Project 1 · Prepared 3 November 2026*

---

## Contents

1. [The dataset in one paragraph](#1-the-dataset-in-one-paragraph)
2. [Where the data comes from (the original paper)](#2-where-the-data-comes-from)
3. [The Hugging Face version: fields, splits, labels](#3-the-hugging-face-version)
4. [What the text actually looks like (quirks and noise)](#4-what-the-text-actually-looks-like)
5. [The task(s) you can define on it](#5-the-tasks-you-can-define)
6. [The fairness side: gender–occupation pairs](#6-the-fairness-side)
7. [Key findings from the literature](#7-key-findings-from-the-literature)
8. [What Project 1 asks for, mapped onto this dataset](#8-what-project-1-asks-for)
9. [Tokenisation for this dataset](#9-tokenisation)
10. [Representation spaces (features)](#10-representation-spaces)
11. [Architectures](#11-architectures)
12. [Evaluation: metrics](#12-evaluation-metrics)
13. [Cross-validation on this dataset](#13-cross-validation)
14. [Bayesian estimation and the ROPE](#14-bayesian-estimation-and-the-rope)
15. [Compute considerations](#15-compute-considerations)
16. [The five backup slides — what each one covers](#16-the-five-backup-slides)
17. [Concepts the oral assessment is likely to probe](#17-concepts-the-oral-is-likely-to-probe)
18. [Ethical and practical caveats](#18-ethical-and-practical-caveats)
19. [Glossary](#19-glossary)
20. [References](#20-references)

---

## 1. The dataset in one paragraph

Bias in Bios is a collection of roughly 400,000 short, English-language, third-person **online professional biographies** scraped from the web. Each biography is labelled with the person's **occupation** (one of 28) and a **binary gender** (inferred from pronouns). The classification task is to predict the occupation from the biography. What makes it special is that gender is a *protected attribute* attached to every example, so you can measure not just *how well* a classifier predicts occupations, but *for whom* it predicts them well — e.g. whether it is better at recognising male surgeons than female surgeons. It was built to study how automated hiring/recruiting tools can reproduce and amplify existing gender imbalances.

---

## 2. Where the data comes from

**Paper:** De-Arteaga et al., *Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting*, ACM FAT\* 2019 (authors from CMU, Microsoft Research, LinkedIn, UMass Lowell).

### 2.1 How it was built

- **Source:** Common Crawl — 16 crawls between 2014 and 2018 (the "WET" files, i.e. cleaned text extracted from web pages).
- **Finding biographies:** they kept lines that start with a name-like pattern (two capitalised words) followed by **"is a(n) … *title*"**, where *title* is an occupation from the US Bureau of Labor Statistics (BLS) Standard Occupation Classification.
- **Choosing occupations:** the 28 most frequent occupations in a sample were selected. Some were merged — e.g. "economics professor", "history professor", … all became **professor**.
- **De-duplication:** bios with the same first name, last name and occupation (with compatible middle names) were treated as duplicates.
- **Result:** 397,340 biographies. Professor was by far the largest class (118,400 bios); rapper the smallest (1,406). Bio length ranged from 18 to 194 tokens, median 72.
- **Original splits:** stratified by occupation — 65% train, 10% validation, 25% test.

### 2.2 How the labels were obtained

- **Occupation label = the title in the first sentence.** "*Nancy Lee is a registered nurse.* She graduated from …" → label `nurse`.
- **The first sentence is then removed** from the input. The model must predict `nurse` from *"She graduated from Lehigh University … years of experience in weight loss surgery, patient support…"*. This is why the HF text column is called **`hard_text`**: the easy giveaway sentence is gone.
- **Gender** was inferred from the gendered pronouns in the bio (he/she…). The authors explicitly note this is a **binary simplification** that erases people who don't fit it, and that it reflects *likely self-identified* gender only insofar as bios are written by/for their subjects.

### 2.3 The experimental design of the paper (useful because it mirrors your course)

The paper compared **three representations** for 28-way occupation classification:

| Paper's name | What it is | Classifier | Link to your tutorial |
|---|---|---|---|
| **BOW** | Binary vector: 1 if a word type occurs in the bio, else 0 | One-vs-all logistic regression, L2 | Essentially the tutorial's **token-indicator** encoder (but with words, not BPE units) |
| **WE** | Average of pretrained fastText word embeddings | One-vs-all logistic regression, L2 | A natural **"third representation"** candidate |
| **DNN** | Bi-directional GRU + attention over fastText embeddings, trained end-to-end | Softmax output | A sequence model (beyond Project 1 scope, relevant later in the course) |

Each was run in **two scenarios**:
1. **With** explicit gender indicators (the raw bio).
2. **"Scrubbed"** — the subject's first name and the words *he, she, her, his, him, hers, himself, herself, mr, mrs, ms* deleted.

---

## 3. The Hugging Face version

### 3.1 Which version this is

The `LabHC/bias_in_bios` upload (Laboratoire Hubert Curien, MIT licence) is the version used by **Ravfogel et al. (2020)** in *Null It Out* (the INLP paper). It is slightly smaller than the original because some biographies could no longer be retrieved from the web. It is the de-facto standard version in the debiasing literature.

### 3.2 Columns

| Column | Type | Meaning |
|---|---|---|
| `hard_text` | string (≈150–967 characters) | The biography **with the first (occupation-revealing) sentence removed**. Pronouns and names are **still present** (i.e. this is the *unscrubbed* setting). |
| `profession` | int, 0–27 | Occupation label (mapping below) |
| `gender` | int, 0–1 | **0 = male, 1 = female** |

### 3.3 Splits

| Split | Rows (≈) |
|---|---|
| `train` | 257,000 |
| `dev` | 39,600 |
| `test` | 99,100 |
| **Total** | **396,189** |

Label and gender distributions are reported to be similar across the three splits.

```python
from datasets import load_dataset
train = load_dataset("LabHC/bias_in_bios", split="train")
dev   = load_dataset("LabHC/bias_in_bios", split="dev")
test  = load_dataset("LabHC/bias_in_bios", split="test")
```

### 3.4 The 28 occupations and their frequencies

| id | profession | % | | id | profession | % |
|---|---|---|---|---|---|---|
| 0 | accountant | 1.42 | | 14 | painter | 1.95 |
| 1 | architect | 2.55 | | 15 | paralegal | 0.45 |
| 2 | attorney | 8.22 | | 16 | pastor | 0.64 |
| 3 | chiropractor | 0.67 | | 17 | personal_trainer | 0.36 |
| 4 | comedian | 0.71 | | 18 | photographer | 6.13 |
| 5 | composer | 1.41 | | 19 | physician | 10.35 |
| 6 | dentist | 3.68 | | 20 | poet | 1.77 |
| 7 | dietitian | 1.00 | | 21 | **professor** | **29.80** |
| 8 | dj | 0.38 | | 22 | psychologist | 4.64 |
| 9 | filmmaker | 1.77 | | 23 | rapper | 0.35 |
| 10 | interior_designer | 0.37 | | 24 | software_engineer | 1.74 |
| 11 | journalist | 5.03 | | 25 | surgeon | 3.43 |
| 12 | model | 1.89 | | 26 | teacher | 4.09 |
| 13 | nurse | 4.78 | | 27 | yoga_teacher | 0.42 |

> Note: the table on the HF dataset card accidentally repeats "nurse 13 / painter 14" in its left column. The correct mapping is the clean 0–27 list above.

```python
NAMES = ["accountant","architect","attorney","chiropractor","comedian","composer",
         "dentist","dietitian","dj","filmmaker","interior_designer","journalist",
         "model","nurse","painter","paralegal","pastor","personal_trainer",
         "photographer","physician","poet","professor","psychologist","rapper",
         "software_engineer","surgeon","teacher","yoga_teacher"]
```

### 3.5 What the distribution implies

- **Severe class imbalance.** Professor alone is ~30% of the data. The smallest classes (rapper, personal_trainer, interior_designer, dj, yoga_teacher, paralegal) are each under 0.5%. The ratio between the largest and smallest class is roughly **85 : 1**.
- **Majority-class baseline ≈ 29.8% accuracy** (always predict "professor"). Any accuracy number must be read against this, not against 1/28 ≈ 3.6%.
- **Head vs tail.** The top 5 classes (professor, physician, attorney, photographer, journalist) cover ~60% of examples. A model can achieve high accuracy while being bad at most of the 28 classes.
- **Gender overall:** 53.9% male, 46.1% female — nearly balanced *overall*, but **highly imbalanced within most occupations** (see §6). The overall balance hides the structure that matters.

---

## 4. What the text actually looks like

Looking at real rows from the training split reveals several properties that matter for every design decision.

### 4.1 Explicit gender signals are everywhere

Almost every bio starts with or contains *He/She/His/Her*, and often the first name ("Brittni graduated…", "Steve has practiced law…"). In this unscrubbed version, a bag-of-tokens model **sees gender directly**. With BPE, names are split into sub-word pieces that can still carry gender signal.

### 4.2 Many bios are templated / boilerplate

The web is full of auto-generated profile pages, and they show up here:

- **Medical-directory templates:** "*Dr. X accepts Medicare insurance… speaks Spanish… is professionally affiliated with…*", "*Ms. X's NPI Number is #… and has been listed in the NPI registry for N years…*", "*He takes Blue Cross Blue Shield…*".
- **Healthcare-listing templates:** "*She graduated with honors in 2005. Having more than 11 years of diverse experiences, especially in NURSE PRACTITIONER, …*" — note this sometimes **leaks the label** in capitals (NURSE PRACTITIONER for nurse, CLINICAL PSYCHOLOGIST for psychologist).
- **Rating-site text:** "*… ratings are above the average of 4.25 stars…*" (a teacher example).

Consequence: some classes are partly identifiable by *template* rather than by *content*, which inflates their scores and can make error analysis interesting (e.g. physician vs nurse vs psychologist share templates).

### 4.3 Noise and personal data

- Contact info: e-mail addresses, phone numbers, street addresses, NPI numbers.
- Scraping debris: "*Downloads/Updates*", "*Bestsellers in this subject*", run-together sentences ("…Artist of the Month.Alexander Sadoyan").
- Occasionally **multiple people** in one bio (e.g. a poet example that mentions other writers), or text that is not really a biography (a paragraph about Alderfer's motivation theory labelled psychologist).

### 4.4 Label ambiguity and semantic overlap

The label is whatever title appeared in the removed first sentence, so the remaining text can be genuinely ambiguous:

| Confusable group | Why |
|---|---|
| professor ↔ everything academic (psychologist, surgeon, physician, software_engineer, dietitian, poet…) | "Professor" was merged from "*X professor*" titles; a professor of surgery writes like a surgeon |
| physician ↔ surgeon ↔ nurse ↔ chiropractor ↔ dentist | Shared medical vocabulary and templates |
| attorney ↔ paralegal | Same legal vocabulary; paralegal is tiny |
| teacher ↔ professor ↔ yoga_teacher ↔ personal_trainer | "teaches", "classes", "students" |
| painter ↔ photographer ↔ architect ↔ interior_designer | Art/design vocabulary |
| composer ↔ dj ↔ rapper ↔ comedian | Music/performance vocabulary |
| filmmaker ↔ journalist ↔ photographer ↔ model | Media vocabulary |
| poet ↔ journalist ↔ professor | Writing/publication vocabulary |

These confusions are where gender can act as a "tie-breaker" for a model — which is exactly how bias appears (see §6–7).

### 4.5 Length

`hard_text` ranges from about 150 to about 967 characters; the original paper reports a median of ~72 tokens. That is longer than AG News items in the tutorial, but still short: in a single bio **most word types occur once**, which matters for the indicator-vs-counts comparison (§10).

---

## 5. The tasks you can define

| Task | Target | Classes | Status |
|---|---|---|---|
| **Occupation classification** | `profession` | 28 | **The standard task.** Gender is used only for *evaluation*. This is what the course description refers to. |
| Occupation classification, gender-scrubbed input | `profession` | 28 | Same task, different input pre-processing; a classic comparison from the original paper. |
| Gender prediction ("probing") | `gender` | 2 | Used in research to show how much gender information leaks through text even after scrubbing. Not a sensible *main* task for this project, and ethically delicate as a product; useful only as a diagnostic. |
| Occupation classification on a subset (e.g. medical or arts occupations) | `profession` | fewer | Possible way to make experiments cheaper or a confusion-pair study sharper, at the cost of comparability with the literature. |

**The course framing** — *"evaluate whether models are more/less successful at selecting the correct occupation for certain gender–occupation pairs"* — means: train a normal 28-way occupation classifier, then break its performance down by (occupation, gender).

---

## 6. The fairness side

### 6.1 Gender–occupation imbalance

Overall the data is ~46% female, but individual occupations are strongly skewed, mirroring the real labour market (and online self-presentation):

- **Strongly female-dominated:** nurse, dietitian, paralegal, yoga_teacher, interior_designer, model, (teacher and psychologist lean female).
- **Strongly male-dominated:** surgeon (only **14.6%** women in the original test split), software_engineer, rapper, dj, composer, pastor, comedian, architect, chiropractor.
- **Roughly balanced:** physician (about half female), journalist, poet, professor, painter.

Exact per-occupation percentages are not on the dataset card; they come from cross-tabulating `profession × gender` in the data.

### 6.2 Why this produces bias in a classifier

A model that maximises accuracy learns *whatever* features predict occupation — including gender words. If 90% of nurses are women, "she" becomes evidence for *nurse*. Then a **male** nurse's bio is pushed toward physician/other, and a **female** surgeon's toward physician/nurse. The classifier is *less accurate for the minority gender within each occupation*.

### 6.3 The core fairness quantities

Let $Y$ be the true occupation, $\hat Y$ the prediction, $G$ the gender.

**True positive rate (recall) per gender and occupation:**
$$\mathrm{TPR}_{g,y} = P(\hat Y = y \mid G = g,\ Y = y)$$

**TPR gender gap** for occupation $y$:
$$\mathrm{Gap}_{y} = \mathrm{TPR}_{\text{female},y} - \mathrm{TPR}_{\text{male},y}$$
Positive → the model recognises women in that occupation better than men; negative → the reverse. This is closely related to the **equality of opportunity** criterion (Hardt et al., 2016): equal TPRs across groups.

**Summary over occupations — RMS TPR gap** (used by Ravfogel et al. and most follow-up work):
$$\mathrm{Gap}^{\mathrm{RMS}} = \sqrt{\frac{1}{|C|}\sum_{y\in C} \mathrm{Gap}_y^2}$$

**Correlation with representation:** Pearson correlation between $\mathrm{Gap}_y$ and $\pi_{\text{female},y} = P(G=\text{female}\mid Y=y)$ across the 28 occupations. A high positive correlation means "the model favours whichever gender already dominates the occupation".

**Compounding imbalance (paper's Theorem 1).** If a gender is under-represented in an occupation ($\pi_{g,y}<0.5$) *and* has a lower TPR ($\mathrm{Gap}<0$ for that gender), then among the people the classifier correctly identifies, that gender is *even more* under-represented than in the population:
$$P(G=g \mid Y=\hat Y=y) < \pi_{g,y}.$$
Surgeon example from the paper: 14.6% of surgeons are women; the BOW model recognised 71% of male surgeons but only 54.5% of female surgeons, so only **11.6%** of the correctly identified surgeons were women. If such a system fed a hiring pipeline repeatedly, the imbalance would grow ("leaky pipeline").

### 6.4 Other fairness views (for context)

| Metric family | Question it asks |
|---|---|
| TPR gap (equal opportunity) | Of the real *y*'s, does each gender get recognised equally often? |
| FPR gap | Do non-*y*'s get wrongly labelled *y* at different rates per gender? |
| Precision (PPV) gap | When the model says *y*, is it equally often right per gender? |
| Demographic parity | Is *y* predicted equally often for each gender, regardless of truth? |

These can disagree with each other (it is mathematically impossible to satisfy all of them at once when base rates differ), which is why papers state which notion they use.

### 6.5 Counterfactual gender-swap test

Swap gender indicators in each bio (he↔she, his↔her, Mr↔Ms…, remove first names) and re-predict. The **flip rate** — fraction of predictions that change — measures how much the model's decision depends on explicit gender words. In the original paper this was 5.5% (BOW), 12.2% (WE), 4.6% (DNN). Note the subtlety the paper raises: this answers "what if this *bio* used the other pronouns", not "what if this *person* were the other gender".

### 6.6 Statistical caveat for small cells

Many gender–occupation cells are tiny (e.g. female rappers, male paralegals, male interior designers). In a single cross-validation fold such a cell may contain only a few dozen examples, so its TPR — and therefore its gap — is a **very noisy estimate**. The RMS gap weights all occupations equally, so noisy small cells can dominate it.

---

## 7. Key findings from the literature

1. **Gender bias exists in all three representations** (BOW, embeddings, GRU). TPR gaps correlate strongly with the occupation's female share: correlations of about **0.82–0.86 with gender indicators**, still **0.71–0.74 after scrubbing**.
2. **Scrubbing reduces the gaps but does not remove them**, and overall accuracy stays roughly the same — a rare example where a fairness intervention costs little performance.
3. **Gender remains predictable from scrubbed text.** On a gender- and occupation-balanced subsample, a BOW gender classifier reached **65.5%** and the DNN **68.2%** (chance = 50%). Words like *women, husband, mother, woman, female* act as **proxies**. The paper's analogy: blind orchestra auditions behind a curtain are not blind if you can still hear the candidate's shoes.
4. **Typical gender-driven confusions** (from the swap experiment): male paralegals predicted as attorneys; male interior designers as architects; female surgeons as physicians; female software engineers and surgeons as professors; female rappers as models; female pastors as teachers.
5. **Follow-up work** (Ravfogel et al. 2020, INLP, and many later debiasing methods) uses this exact HF version, reports **accuracy + RMS TPR gap**, and generally finds a **trade-off**: removing gender information from representations reduces the gap but also some task accuracy.
6. **Metric choice matters.** Later work (e.g. Orgad & Belinkov 2022, *Choose Your Lenses*) shows that different bias metrics can rank the same models differently, and that bias estimates are unreliable when training/evaluation sets are small.

---

## 8. What Project 1 asks for

From `project.ipynb`, mapped onto Bias in Bios:

| Requirement | Concretely for this dataset |
|---|---|
| Compare text-classification systems on your data/task | 28-way occupation classification from `hard_text` |
| **Representations:** token indicators + bag-of-words + **one of your choice** | §10 lists candidates; a scrubbed or TF-IDF / bigram / embedding variant fits naturally |
| **Architectures:** log-linear + single-hidden-layer MLP + **one of your choice** | §11 lists candidates |
| Comparisons based on **cross-validation** and a **relevant metric** | Accuracy is weak here because of imbalance → macro-F1 (and the fairness gap as a secondary metric) — §12 |
| **Bayesian estimation** with a **ROPE** you motivate | Benavoli correlated-t posterior over fold-wise differences — §14 |
| Oral assessment, up to **5 backup slides**: data & metrics, features, architectures, results, extra | §16 |
| Same dataset carries into Projects 2 and 3 and the final blog post | Choices (tokeniser, splits, metric) should be ones you're happy to keep |
| Reproducible, organised code; record AI assistance | Seeds, saved splits, saved tokenizer, a log of what was delegated |

---

## 9. Tokenisation

### 9.1 What data to train the BPE tokenizer on

- **This dataset is large enough on its own.** ~257k training bios (≈ 18–19M tokens at a median of ~72 tokens each) is about twice AG News in example count, so the "dataset too small to train a tokeniser" concern in the project notebook does **not** apply.
- **Train it on the `train` split only** — never on `dev`/`test`, to keep held-out text unseen. (Inside cross-validation, a tokenizer fitted on the whole training split has technically seen the validation folds' raw text; because BPE is unsupervised and never sees labels, this is usually considered a negligible leak, but it is a defensible point to mention.)
- **Reusing the AG News tokenizer** is possible but involves a **domain shift** (news vs professional bios): words like *residency, dissertation, chiropractic, NPI* would be split into many small pieces. This is exactly tutorial question 4 ("tokenizer and classifier trained on different data").

### 9.2 Hyperparameters and their effects here

| Hyperparameter | Effect on Bias in Bios |
|---|---|
| `vocab_size` | Small (e.g. 1k, the tutorial default) → common words survive but occupation-specific terms (*dermatologist, paralegal, choreographer*) get fragmented into less informative pieces. Larger (e.g. 5k–30k) → more whole words, more informative features, larger (sparser) feature space, more parameters to fit. |
| `min_frequency` | Higher values stop rare merges (rare names, institution names, phone-number fragments) from entering the vocabulary. |
| Lowercasing / normalisation | Merges "He"/"he", "Professor"/"professor"; reduces sparsity; loses the ALL-CAPS template cue (NURSE PRACTITIONER). |
| Pre-tokenisation | Metaspace (tutorial default) treats whitespace as boundary; punctuation and numbers (phone numbers, years) still produce many tokens. |

### 9.3 BPE and gender

With BPE, a first name like "Brittni" becomes sub-word pieces; pieces like *-ni*, *-a* or *-ette* may correlate with gender. So "scrubbing" pronouns at the word level does not fully remove name-based signal unless names are removed **before** tokenisation. Pronouns (*he, she, his, her*) are frequent enough to become single BPE tokens.

---

## 10. Representation spaces

Every representation maps a bio $x$ to a vector $\mathbf h(x)\in\mathbb R^D$.

### 10.1 Required: token indicators

$h_v(x) = 1$ if BPE token $v$ occurs in $x$, else 0. Dimension $D = |V|$. This is essentially the paper's BOW representation (they used words instead of BPE units). It ignores frequency and order.

### 10.2 Required: bag-of-words (counts)

$h_v(x) = $ number of times token $v$ occurs in $x$. Same dimension, same sparsity pattern; adds repetition information.

**What to expect on this dataset:** bios are short, so most *word-level* tokens appear once and counts ≈ indicators. With a *small* BPE vocabulary, sub-word pieces repeat more, so counts differ more. Counts are also **length-sensitive**: longer bios get larger vectors, which interacts with regularisation and optimisation (log-linear models are not scale-invariant). The tutorial observed a small gain from counts on AG News; on Bias in Bios the difference may be within any sensible ROPE — which is itself a perfectly reportable finding.

### 10.3 Candidates for the third representation

| Option | What it is | Why it is interesting here | Cost / caveat |
|---|---|---|---|
| **TF-IDF** | Counts re-weighted by inverse document frequency (and usually L2-normalised) | Down-weights ubiquitous tokens (*the, he, his, university*) and up-weights distinctive ones (*chiropractic, choreograph*); normalisation removes length effects | Classic, cheap, sparse; IDF must be fitted on training folds only |
| **Normalised / sublinear frequencies** | $\log(1+\text{count})$ or count / length | Removes the length bias of raw counts; cheap ablation | Small conceptual step from BoW |
| **Bag of bigrams (+ unigrams)** | Counts of adjacent token pairs | Captures phrases: *"law school"*, *"medical school"*, *"yoga classes"*, *"personal training"*, *"NPI number"* | Dimension explodes ($|V|^2$ possible); needs a frequency cutoff or hashing |
| **Gender-scrubbed BoW** | BoW after deleting pronouns/titles (and ideally names) before tokenisation | Directly tests the paper's central question on your own models: does removing explicit gender change accuracy and the TPR gap? Strongest "story" for the dataset | It is a *pre-processing* change; argue clearly why it counts as a representation choice (it changes the feature space by removing coordinates) |
| **Averaged pretrained word embeddings** (fastText/GloVe) | Mean of word vectors → dense ~300-d vector | Same as the paper's WE; brings in external semantic knowledge; tiny input dimension | External resource; embeddings themselves carry known gender biases; loses token identity |
| **Character n-grams** | Counts of character 3–5-grams | Robust to typos, morphology, rare words | Large sparse space; overlaps conceptually with BPE |
| **Hashed features** | Hash any of the above into a fixed number of buckets | Makes bigrams/char-n-grams tractable | Collisions make features uninterpretable |

### 10.4 How representation interacts with bias

Any representation that keeps pronoun/name coordinates gives the classifier direct access to gender. Representations that change *weighting* (TF-IDF) change how much those coordinates matter: pronouns occur in nearly every bio, so their IDF is low — TF-IDF implicitly down-weights *he/she*, which may or may not reduce the gap.

---

## 11. Architectures

### 11.1 Required: log-linear softmax regression

$$p(y\mid x) = \mathrm{softmax}(\mathbf W\mathbf h(x) + \mathbf b),\quad \mathbf W\in\mathbb R^{28\times D}$$

- One weight vector per occupation; **weights are directly interpretable**: the largest weights for *nurse* tell you which tokens push towards nurse — including, possibly, *she*.
- Convex training objective; fast with sparse inputs (scikit-learn `LogisticRegression`, as in `part1.loglin`).
- Regularisation strength (`C` in scikit-learn) matters with large sparse $D$.
- It can only draw linear decision boundaries in feature space: it cannot represent interactions like "*she* AND *surgery*".

### 11.2 Required: single-hidden-layer MLP

$$p(y\mid x) = \mathrm{softmax}(\mathbf W_2\,\mathrm{ReLU}(\mathbf W_1\mathbf h(x)+\mathbf b_1)+\mathbf b_2)$$

- Can model feature interactions (e.g. "*surgery* without *nurse*").
- **Useful insight:** when $\mathbf h(x)$ is a count vector, $\mathbf W_1\mathbf h(x) = \sum_{v} \text{count}_v(x)\,\mathbf W_1[:,v]$ — the first layer is literally a **sum of learned token embeddings** (an "embedding bag"). With indicators it is a sum over distinct tokens. So the MLP on BoW is already a simple neural bag-of-embeddings model.
- Non-convex; results depend on seed, learning rate, epochs, dropout, weight decay — more variance across folds.

### 11.3 Candidates for the third architecture

| Option | Description | What it tests |
|---|---|---|
| **Deeper MLP** (2 hidden layers) | Add a layer; keep width comparable | Does extra non-linearity help on 28 overlapping classes? |
| **Wider / different activation / dropout** | e.g. hidden 512, GELU or tanh, dropout 0.3–0.5 | Capacity vs. overfitting; dropout as regulariser for many sparse inputs |
| **Class-weighted loss** (logistic or MLP) | Cross-entropy weighted inversely to class frequency | Directly targets imbalance; usually trades accuracy for macro-F1; may also change gender gaps for small classes. Strictly it is a training-objective change, not an architecture change — worth stating explicitly if used. |
| **Mean-pooled / normalised embedding-bag** | Divide the first-layer sum by bio length (average instead of sum) | Removes length sensitivity; becomes a fastText-style classifier |
| **Multinomial Naive Bayes** | Generative: $p(y)\prod_v p(v\mid y)^{\text{count}}$ | A different modelling philosophy (generative vs discriminative); very fast; strong baseline for text |
| **Linear SVM** | Hinge loss instead of log-loss | Margin-based alternative; no calibrated probabilities (matters for ensembling/pmf-based decisions) |

### 11.4 How architecture interacts with bias

A linear model can only add up per-token evidence: *she* contributes the same push towards *nurse* in every bio. An MLP can learn interactions, which in principle allows it to use gender only when content is ambiguous — or to entangle gender more deeply. Which happens is an empirical question, measurable via the TPR gap.

---

## 12. Evaluation metrics

### 12.1 Why accuracy alone is inadequate here

- Majority baseline is ~30%; the top 5 classes are ~60% of data.
- Accuracy is dominated by professor/physician/attorney performance; a model can ignore rapper, dj, paralegal, personal_trainer, interior_designer and yoga_teacher and lose < 2.5% accuracy.
- Accuracy is gender-blind; it cannot show *for whom* the model fails.

(This is precisely tutorial question 7: "use your labelled data to defend or challenge accuracy".)

### 12.2 Alternatives

| Metric | Definition | Pros | Cons |
|---|---|---|---|
| **Accuracy** | fraction correct | Comparable to literature (Ravfogel et al. report it); intuitive | Dominated by large classes |
| **Macro-F1** | unweighted mean of per-class F1 | Every occupation counts equally; sensitive to minority classes | Noisier across folds (small classes) → wider posteriors |
| **Balanced accuracy** | mean of per-class recall (TPR) | Directly the average of the TPRs the fairness analysis uses | Ignores precision |
| **Weighted-F1** | per-class F1 weighted by support | Between the two | Behaves similarly to accuracy here |
| **Per-class precision/recall + confusion matrix** | — | Reveals *which* occupations get confused (§4.4) | Not a single number for Bayesian comparison |
| **RMS TPR gender gap** | §6.3 | The dataset's raison d'être; standard in literature | Not a performance metric; noisy for small cells; lower is "fairer" |
| **Correlation of $\mathrm{Gap}_y$ with $\pi_{\text{female},y}$** | §6.3 | Captures *compounding* specifically | Only 28 points; says nothing about magnitude |

A common reporting pattern is: one **primary performance metric** for the Bayesian comparison (e.g. macro-F1, with accuracy reported alongside for comparability), plus **fairness metrics** as a secondary analysis (which can *also* be compared across systems with the same Bayesian machinery, since they are computed per fold).

---

## 13. Cross-validation

### 13.1 What the tutorial helper does

`part1.experiment.cross_validation_split(pairs, k, seed)` takes `(text, label)` pairs and returns `k` folds of `{"training", "validation"}`. Each system is then trained on each fold's training part and scored on its validation part, giving **k scores per system** (or `k × r` with `r` seeds).

### 13.2 Considerations specific to Bias in Bios

- **Gender must survive the split.** The tutorial pairs only text and label. For fairness metrics, the gender of each validation example has to be retrievable (e.g. by keeping indices, or splitting triples).
- **Stratification.** With classes as small as 0.35%, purely random folds have slightly varying class counts; stratifying by occupation (or by occupation × gender) keeps folds comparable. Whether the tutorial helper stratifies can be read from its source.
- **Identical folds across systems.** The Bayesian correlated-t model analyses *paired* differences, fold by fold. Every system must be evaluated on exactly the same folds (same seed, same split object).
- **Where the official splits fit.** The dataset already provides `dev` and `test`. Cross-validation runs over `train` (for model comparison); `test` stays untouched; `dev` can play the role of the "devtest" set the tutorial mentions (e.g. for choosing an ensemble or decision rule).
- **Folds vs fold size.** Each validation fold of a 5-fold split of train holds ~51k bios; that is large enough for stable accuracy, but the smallest gender–occupation cells still have few examples per fold (§6.6).

---

## 14. Bayesian estimation and the ROPE

### 14.1 The model (Benavoli et al. 2017, "Bayesian correlated t-test")

Data: the vector of fold-wise differences $m_i = \text{score}_B^{(i)} - \text{score}_A^{(i)}$ for $N = k\times r$ folds.

- $\mathbf m \sim \mathrm{MVN}(\mathbf 1\mu,\ \boldsymbol\Sigma)$, with $\Sigma_{ii}=\sigma^2$ and $\Sigma_{ij} = \rho\sigma^2$ (off-diagonal).
- $\rho$ is **not learned**; it is set by a heuristic: $\rho = n_{\text{test}}/n_{\text{total}} = 1/k$ for k-fold CV. It accounts for the fact that CV folds share most of their training data, so their scores are correlated; ignoring this would make the posterior over-confident.
- The posterior over the mean difference $\mu$ is a Student-t; `part1.analysis.BenavoliCorrelatedT` computes it.

### 14.2 Reading the output

Given a ROPE half-width $r$:
- $P(\mu < -r)$ → A practically better
- $P(-r \le \mu \le r)$ → practically equivalent
- $P(\mu > r)$ → B practically better
- **Effect size** = posterior mean difference / posterior std: how separated the systems are relative to fold-to-fold variability.

This is *estimation*, not a test: it tells you how much posterior belief lands in each region; the decision remains yours.

### 14.3 Choosing and motivating the ROPE for this dataset

The ROPE must be in the **units of the metric**:

- **Accuracy:** the tutorial's 0.005 (half a percentage point) follows NLP-paper convention. On ~51k-example folds, 0.005 ≈ 255 more bios correctly classified — a meaningful but small difference.
- **Macro-F1:** a single small class can swing macro-F1 a lot. 0.005 in macro-F1 is a much smaller effect "per bio" than in accuracy (it can be driven by a handful of rapper/dj examples). Arguments exist for a somewhat larger ROPE (e.g. 0.01), justified by fold-to-fold noise or by "what change would a user of an occupation classifier notice?".
- **RMS TPR gap:** units are TPR differences. A practitioner-oriented argument: a gap reduction smaller than ~1 percentage point is unlikely to matter for any individual occupation's hiring pipeline. Literature differences between debiasing methods are often several points.
- Whatever the value, the defence in the oral is the *reasoning* (domain convention, fold noise, practical consequence), not the number.

### 14.4 Effect of the number of folds and seeds

More folds/seeds → more observations → narrower posterior. But because of the $\rho$ correction, simply increasing $k$ gives diminishing returns; repeating CV with **different seeds** (`r > 1`) adds more independent information. Benavoli et al. recommend repeated CV (e.g. 10 runs of 10-fold) when affordable.

### 14.5 Pairwise comparisons

With 3 representations × 3 architectures = **9 systems**, there are 36 possible pairs. Not all pairs are equally informative; the most interpretable comparisons hold one factor fixed (e.g. three representations under the same architecture; three architectures under the same representation).

---

## 15. Compute considerations

- **Scale vs AG News.** The tutorial's AG News train split is 120k items; one 5-fold run took ~3 min (logistic) and ~20 min (MLP, 10 epochs, CPU). Bias in Bios train is ~2.1× larger and texts are longer, so expect roughly **2–3× those times per configuration**.
- **Full grid:** 9 systems × 5 folds × (number of seeds) — MLP-based systems dominate the cost.
- **Ways people reduce cost:** a stratified **subsample** of train for CV (e.g. 50–100k), fewer epochs with early stopping on the fold's validation part, larger batch sizes, GPU if available, caching encoded sparse matrices so encoding is done once per tokenizer/representation rather than per fold.
- **Trade-off of subsampling:** smaller folds → smaller rare cells → noisier macro-F1 and much noisier TPR gaps.
- **Memory:** sparse (`csr_matrix`) representations are essential; a dense 257k × 10k float matrix would be ~10 GB.

---

## 16. The five backup slides

| Slide | What it typically covers for this dataset |
|---|---|
| **Data & metrics** | Dataset origin (Common Crawl bios, first sentence removed), 28 classes, imbalance (professor ~30%, majority baseline), gender skew per occupation, splits used, chosen metric(s) and why accuracy is insufficient, ROPE value and its justification |
| **Features** | Tokenizer choice (data, vocab size, min frequency, normalisation), the three representation spaces, dimensionality and sparsity, what each captures/ignores, how gender indicators enter each |
| **Architectures** | Log-linear, MLP, third model; parameter counts; hyperparameters; why the third one was chosen |
| **Results** | Box/point plots across folds, posterior plots with ROPE, P(A)/P(tie)/P(B) and effect sizes for key pairs, best system |
| **Extra** | Fairness analysis: per-occupation TPR gaps vs % female, RMS gap per system, confusion pairs, scrubbing comparison, error analysis examples, data quirks (templates, leakage) |

---

## 17. Concepts the oral is likely to probe

The project notebook says the TA asks *what* you did and *why* (vs reasonable alternatives), and pros/cons. For this dataset, the "why" questions naturally include:

- **Why this metric?** Imbalance argument; what accuracy hides; what macro-F1 costs (noise).
- **Why this tokenizer and vocabulary size?** Data size, domain match, fragmentation of occupation words, effect of vocab size on informativeness (tutorial BPE questions 1–3).
- **Indicators vs counts — what changes and why might it (not) matter for short bios?**
- **Why is the third representation/architecture a reasonable choice and what did you expect?**
- **What does the MLP's first layer compute on sparse input?** (sum of token embeddings, §11.2)
- **Why cross-validation and not a single dev split? Why must folds be shared across systems?**
- **What is the ROPE, why that size, what is the effect size, why not a p-value?** (Tutorial's NHST-vs-estimation discussion.)
- **What is $\rho$ in the correlated-t model and why is it needed?**
- **How does gender enter the model, and how would you measure that?** (TPR gap, swap test, inspecting logistic weights for *he/she*.)
- **Limitations:** binary inferred gender, label noise, templates, web-scraped population ≠ workforce.
- Tutorial-general questions: Unicode/encodings, Heaps' law, BPE algorithm, tabular CPDs and their limits, softmax regression, decision rules (argmax and when it is counter-intuitive), ensembling.

---

## 18. Ethical and practical caveats

- **Binary, inferred gender.** Gender is derived from pronouns; non-binary people are erased, and misgendering is possible. Results speak about *pronoun-marked groups*, not about gender identity.
- **Personal data.** Real names, phone numbers, e-mails and addresses scraped from the web appear in the text. Quoting raw examples in a public blog post deserves care (e.g. redacting names/contacts).
- **Population.** People with online bios differ from the workforce; occupations are US-BLS-centric; text is English-only.
- **Label noise.** Labels come from a regex on the removed first sentence; some bios are templates, multi-person pages, or off-topic.
- **Purpose.** The dataset exists to *study* bias in hiring-like systems. Building a gender predictor as a product would be a misuse; gender prediction is used in research only as a diagnostic of information leakage.
- **Licence.** MIT (original Microsoft `biosbias` repository and this HF upload).

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **`hard_text`** | Bio with the occupation-revealing first sentence removed |
| **Scrubbing** | Deleting explicit gender indicators (first name, pronouns, titles) from the input |
| **Explicit gender indicator** | A token that directly reveals gender (he/she, Mr/Ms, first name) |
| **Proxy** | A non-explicit token correlated with gender (e.g. *husband*, *mother*, *women*) |
| **TPR (recall)** | Fraction of true members of a class that are predicted as that class |
| **TPR gender gap** | $\mathrm{TPR}_{\text{female},y} - \mathrm{TPR}_{\text{male},y}$ |
| **RMS gap** | Root-mean-square of per-occupation gaps — a single bias number |
| **Equality of opportunity** | Fairness criterion requiring equal TPRs across groups |
| **Compounding imbalance** | A classifier making an already under-represented group even rarer among its positive predictions |
| **Allocation harm** | Harm from unequal distribution of opportunities/resources (here: being found by a recruiter) |
| **ROPE** | Region of practical equivalence: differences smaller than this are treated as "no practical difference" |
| **Correlated t / Benavoli** | Bayesian model of fold-wise score differences that accounts for CV correlation via $\rho$ |
| **Effect size** | Posterior mean difference divided by posterior standard deviation |
| **INLP** | Iterative Nullspace Projection (Ravfogel et al. 2020), a debiasing method; source of this HF version |
| **Macro-F1** | Unweighted average of per-class F1 scores |

---

## 20. References

- De-Arteaga, M., Romanov, A., Wallach, H., Chayes, J., Borgs, C., Chouldechova, A., Geyik, S., Kenthapadi, K., Kalai, A. T. (2019). *Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting.* FAT\* '19. [arXiv:1901.09451](https://arxiv.org/abs/1901.09451) · [doi:10.1145/3287560.3287572](https://doi.org/10.1145/3287560.3287572)
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020.
- Hugging Face dataset card: [LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios)
- Original data-recreation code: [github.com/microsoft/biosbias](https://github.com/microsoft/biosbias)
- Romanov, A. et al. (2019). *What's in a Name? Reducing Bias in Bios without Access to Protected Attributes.* NAACL 2019. [arXiv:1904.05233](https://arxiv.org/abs/1904.05233) — defines the RMS TPR gap.
- Hardt, M., Price, E., Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
- Orgad, H., Belinkov, Y. (2022). *Choose Your Lenses: Flaws in Gender Bias Evaluation.* [arXiv:2210.11471](https://arxiv.org/abs/2210.11471)
- Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.* JMLR 18. (Section 3: correlated t.)
- Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL. (BPE.)
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Ch. 2 (tokenisation, BPE) and Ch. 4 (logistic regression).
- NLP1 Project 1 and Tutorial 1 notebooks (UvA, 2026).

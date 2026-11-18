# Oral prep: likely TA questions with short answers

All numbers come from `notebooks/bias-in-bios-part1.ipynb` (3 folds × 2 seeds, full deduplicated train = 257,081 bios).

## The 30-second story
We predict 28 occupations from bios. **Macro-F1** is the main score, because professor is 30% of the data.
**Best system: MLP + TF-IDF, 0.754.** One hidden layer helps (+0.02–0.03) unless TF-IDF is used; with TF-IDF the
log-linear model is practically as good (0.749). Our deeper MLP (second layer with a shortcut and LayerNorm) matches
the MLP but doesn't beat it. Indicators ≈ counts.
Every system **amplifies gender imbalance**: the recall gap follows % women with r ≈ 0.86. Deleting pronouns costs no
accuracy and cuts the RMS gap by 22%, but doesn't remove it.

## Data & metrics (§1, §6)

**Why macro-F1 and not accuracy?**
"Always professor" gets 0.298 accuracy but 0.016 macro-F1. Macro-F1 averages F1 over the 28 occupations, so rare
ones count equally.

**What are precision, recall, F1?**
Precision: when we say *y*, how often are we right? Recall (TPR): of all real *y*, how many do we find?
F1 = 2PR/(P+R).

**Why is balanced accuracy 1/28 for the majority baseline?**
Recall is 1 on professor and 0 on the other 27; the average of the 28 recalls is 1/28.

**Why is random accuracy 0.125?**
It equals Σ pₖ²: the chance that a random guess and a random label agree.

**How was gender obtained?**
It was inferred from pronouns: binary, a proxy, with non-binary people absent. A pronoun rule recovers it in 94.9%
of bios with 100% agreement.

**Data problems?**
- 15 duplicated texts with conflicting labels (label noise); we deduplicated.
- 6.8% of bios share a template opening.
- Web noise is rare (phone numbers 4%).

**TPR gap / RMS gap?**
- gap = recall(women) − recall(men) per occupation.
- RMS = √(mean of gap²) over occupations (Romanov et al., 2019).
- Computed on out-of-fold predictions combined over folds: all 88 female rappers instead of ~29 per fold.

## Splits (§2)

**Why cross-validation?**
The brief requires it. It gives 6 scores per system for the Bayesian comparison, and every bio is tested once.

**Why 3 folds × 2 seeds, not 5 × 3?**
- Time: ≈ 25–30 min instead of ≈ 1.5 h.
- Each fold still trains on 171k bios.
- The cost is wider posteriors. The Nadeau–Bengio correction multiplies the variance by 1/n + 1/(k−1):
  0.67 for 3×2 vs 0.32 for 5×3.
- 1 seed (3 scores) would be too few to conclude anything.

**Why stratify by occupation × gender?**
So every fold has the same mix: validation parts are 46.11–46.12% women, and the smallest group is 29–30 per fold.

**Why must all systems share the folds?**
The comparison is paired: per split, B − A.

**What about dev/test?**
Untouched this week, kept for later projects.

## Tokenizer (§3)

**How does BPE work?**
Start from characters and repeatedly merge the most frequent adjacent pair, up to 10k units (min. frequency 5).
New text is segmented by replaying the merges in order.

**Why your own tokenizer?**
257k in-domain bios are plenty, and job words should stay whole: 21 of 28 job names are a single token.
A news tokenizer would fragment *chiropractic*, *paralegal* and similar words.

**Is that leakage?**
No labels are involved, and it's identical for all systems (strict alternative: train it on `dev`).

**Your changes?**
Lowercasing, and splitting punctuation off ("engineer." ≠ "engineer" otherwise).

**Effect of vocabulary size?**
Bigger means more whole words and more informative features, but more parameters and sparser counts.

## Representations (§4)

**R1, R2, R3?**
- R1: 1/0, is the token there?
- R2: the count.
- R3: TF-IDF = (1 + ln count) · idf, then scaled to length 1, where idf(v) = ln((1+N)/(1+df)) + 1.

**Why TF-IDF?**
- It damps repeats.
- It down-weights common tokens: idf of *the* 1.25, *she* 1.95, *nurse* 4.45, *chiroprac* 7.83.
- It removes bio length.

**Leakage with TF-IDF?**
idf counts bios, so it is learned on each training fold only.

**Can these tell "man bites dog" from "dog bites man"?**
No. They are bags: word order is lost.

**Why sparse?**
Median 57 non-zero slots out of 10,000: 124 MB sparse vs 10.3 GB dense.

**R1 vs R2?**
78% of counts are 1, so we predicted they'd be equal. Result: practically equal for all 3 models (Bayesian
P(equal) ≈ 0.98–0.99).

## Architectures (§5)

**Log-linear?**
softmax(W h + b); 280k weights; a points table where clues add up. It's convex, can't model combinations, and is
easy to inspect.

**MLP?**
128 hidden "detectors" with ReLU, so it can learn combinations. Layer 1 = a sum of learned word vectors
(an embedding bag). 1.28M weights.

**Deeper MLP (ours), why?**
The brief suggests modifying the MLP. We wanted to know whether more combining power separates overlapping jobs.
Design: sparse 10,000 → 128 (LayerNorm, ReLU), then a second 128 → 128 layer whose output is *added* to the first
(residual shortcut, LayerNorm, dropout 0.3, ReZero: it starts switched off), and 20% of each bio's tokens hidden during
training. 1,300,765 weights, only 17k more than the MLP.
Result: practically equal to the MLP on indicators (−0.005) and counts (0.000); unsure on TF-IDF (−0.010). Mean 0.745 vs 0.750.

**Why the redesign? (the story to tell)**
Our first version (plain 256 → 128 stack, 2.6M weights) scored below the MLP (mean 0.735; TF-IDF 0.730 vs 0.754).
Diagnosis: TF-IDF rows have length 1 (counts ≈ 9.7), so the bio's signal reaching layer 2 was tiny next to the
biases. Fixes: LayerNorm (same behaviour whatever the input scale) and a shortcut (it can fall back to the MLP).
That closed most of the gap; token dropout barely mattered (one-split check, [5b+]).

**What do LayerNorm, a residual shortcut and ReZero do?**
LayerNorm rescales each bio's hidden numbers to mean 0 and spread 1. A residual shortcut adds a layer's output to its
input (z₂ = z₁ + f(z₁)), so the extra layer only has to learn a correction. ReZero multiplies that correction by a
learned α that starts at 0, so training starts from exactly the MLP.

**Why does a second layer still not beat one?**
A bag of words has no order or structure for a second layer to exploit beyond the combinations one layer already
learns; both models overfit within 1–3 epochs.

**Early stopping, why an inner hold-out?**
The tutorial picks the best epoch on the validation fold and reports that same fold, which is optimistic. We hold
out 10% of the training part instead.

**Why stop on macro-F1?**
It's our metric; the loss is dominated by big classes (≈ 0.02 worse in a pilot).

**Tuning, what are the candidates?**
- log-linear `C` ∈ {0.3, 1, 3} (small = cautious);
- learning rate ∈ {0.001, 0.003}.

Chosen on a 90/10 split of fold 0's training part. Results:
- `C` = 0.3 for indicators and counts, but 3 for TF-IDF (its values are small, so it needs a weaker penalty);
  `C` = 0.3 on TF-IDF would score 0.69 instead of 0.76;
- learning rate 0.001 for both neural models and all representations (with LayerNorm, the input scale no longer
  matters for the deeper MLP).

**Convergence?**
`loglin_bow` didn't fully converge in 500 iterations (a limitation); the others did.

## Results & Bayesian comparison (§8, §9)

**Best system?**
mlp_tfidf 0.754 ± 0.002. Worst: loglin_bow 0.719. Fold std 0.001–0.004.

**Hardest occupations?**
- Hardest: teacher 0.57, pastor 0.61, architect, interior designer, paralegal (≈ 0.67).
- Easiest: dentist 0.94.
- Main confusions: paralegal → attorney 34%, chiropractor → physician 24%, surgeon → physician 18%.

**What is the posterior, ROPE, HDI?**
- Posterior: how believable each value of the true mean difference is, given the 6 paired differences.
- ROPE: a "too small to matter" band.
- HDI: the narrowest range holding 95% of the belief.
- P(A better) / P(equal) / P(B better) = the areas left of, inside and right of the ROPE.

**Why "correlated"?**
The folds share training data, so their scores aren't independent; the method inflates the variance accordingly
(Nadeau–Bengio).

**Why a ROPE of 0.01?**
- Practical: one macro-F1 point is a change you'd notice and act on.
- It's about 3× the median fold std (0.0031).
- The tutorial uses 0.005 for accuracy, and macro-F1 is noisier.
- Sensitivity: main conclusions hold for 0.005 and 0.02.

**Why not a p-value?**
Estimation communicates *how big* the difference is and how uncertain, and lets us say "practically equal", which
a significance test can't.

**Key verdicts:**
- MLP > log-linear on indicators (+0.021) and counts (+0.028);
- MLP = log-linear on TF-IDF (+0.005, P(equal) 0.998);
- deeper MLP = MLP on indicators and counts; unsure on TF-IDF (−0.010);
- deeper MLP > log-linear on indicators (+0.017) and counts (+0.027);
- indicator = counts everywhere;
- TF-IDF > others for log-linear.

## Fairness (§10, §11)

**Main finding?**
The gap follows % women, r = 0.86 for the best system (0.83–0.90 for all 9).
- model +0.49 (women 0.84 vs men 0.35); dietitian +0.32; paralegal +0.24;
- surgeon −0.21 (women 0.48 vs men 0.68); rapper −0.21.

**Where do the missed bios go?**
- male paralegals → attorney (77% of their misses), male nurses → physician;
- female surgeons → physician, female pastors → teacher, female rappers → model.

This matches De-Arteaga et al.

**Why does it matter?**
Compounding: in a recruiting search the minority gender becomes even rarer among the people found.

**Scrubbing?**
Deleting he/she/his/her/Mr/Ms…:
- macro-F1 0.754 → 0.751 (practically equal, P = 1.00);
- RMS gap 0.149 → 0.116 (−22%), smaller in 23/28 occupations;
- correlation still 0.78.

It leaks through first names, family words and topics.

**Does the model use *she*?**
Log-linear weight(she) − weight(he) follows % women with r = 0.96. The top words are content words, so the pronoun
is a steady secondary nudge.

**Which system is fairest?**
The deeper MLP has the smallest gaps (RMS 0.142 on TF-IDF, 0.149, 0.156), then the MLP (0.149–0.160), then
log-linear (0.159–0.168). Possible reasons: TF-IDF down-weights pronouns, and hiding tokens in training stops the
model leaning on any single word. The differences are small and untested.

**Limitations:**
- binary inferred gender;
- 6 CV scores;
- one knob tuned, on one fold;
- loglin_bow not converged;
- near-duplicate templates across folds;
- names not scrubbed.

## Tutorial questions worth rehearsing
- Heaps' law: V(n) = K·n^β, linear on log–log axes.
- Why tabular CPDs fail: one row per possible text, so they can't generalise to unseen text.
- Unicode vs encodings: a wrong encoding gives an error or mojibake.
- Argmax can be counter-intuitive: near ties, or when errors have unequal costs.
- Ensembles: majority vote; mixture of experts vs product of experts.
- Character-level vs BPE: characters carry little meaning, so a bag of characters loses almost everything.

---

# Part 2: embeddings and compositional encoders

Numbers from `notebooks/bias-in-bios-part2.ipynb`.

## Q1 · Choosing the embedding model (§2)

**Which model and why?**
fastText Common Crawl, 300-d. It has the best coverage (99.3% of running words with our back-off rule), the best
quick task score (0.733 macro-F1 vs 0.721 wiki-news and 0.562 GloVe-50), sensible job neighbours, and it was
trained on Common Crawl, the same source as our bios.

**How did you tokenize, and why not Part 1's BPE?**
The embedding table has one row per *word as its makers cut their text*; BPE pieces (*chiroprac + tor*) wouldn't be
found. We use lowercase + plain quotes + NLTK word tokenization, plus a back-off rule per model.

**What is the back-off rule?**
An unknown hyphen/slash compound is replaced by its known parts (GloVe was trained on split compounds, so it has no
*in-network*); NLTK's quote tokens become a plain quote; *n't* becomes *not*. It raises coverage from
98.2–98.7% to 99.0–99.3%.

**Coverage numbers?**
99.0 / 99.1 / 99.3% of running words (GloVe / wiki-news / Common Crawl), but only 53–59% of *distinct* words: the
missing ones are rare names, institutions and typos (< 1% of the text). Every occupation and gender word is covered.

**Why straighten curly quotes?**
It doesn't change coverage with back-off (99.32 vs 99.33%), but without it NLTK splits *master’s* into
*master* + *’* + *s*, creating junk tokens.

**Are the models compared fairly?**
Not fully: GloVe was 50-d, fastText 300-d, so size and training corpus are confounded. We test embedding size
separately in the embed-and-pool section.

## Q2 · What the embeddings capture (§3)

**Do they capture task-relevant similarity?**
Yes. fastText neighbours of *paralegal*: paralegals, lawyer, litigator, attorney. And occupations close in embedding
space were the ones our Part 1 model confused (Spearman ρ = 0.40–0.56).

**Evidence one model is better than another?**
The quick task test (averaged vectors + log-linear): Common Crawl 0.733, wiki-news 0.721, GloVe-50 0.562 macro-F1.

**Limitations?**
- one vector per word mixes meanings: *model* → prototype, concept (the fashion sense is lost);
- spelling variants dominate (*professor* → prof., dr.);
- 41–47% of distinct words (names) have no vector;
- static: context is ignored;
- built-in social bias.

**What is especially troubling for our task?**
Gender bias. On a *he → she* direction (Bolukbasi et al., 2016), occupation words line up with our data's share of
women: r = 0.90 for Common Crawl. *surgeon − he + she* → *nurse* (GloVe). The embeddings bring a gender skew before
seeing a single bio, which could widen the recall gaps from Part 1.

**What is cosine similarity and why use it?**
How much two vectors point the same way (−1 to 1), ignoring length. Word frequency inflates vector length, so the
direction carries the meaning.

## Q3 · Embed-and-pool (§4): design questions

**What is embed-and-pool?**
Look up the vector of every word in a bio, pool them into one vector (mean or max per dimension), and feed that to a
classifier. Word order is lost, like bag-of-words, but similar words now have similar inputs.

**How did you structure the search?**
In stages, changing one thing at a time. Stage 1: 3 poolings × 3 classifiers with frozen vectors (9 systems). Stage 2:
the winner with trained embeddings (fine-tuned, random start) and smaller sizes (PCA 50/100, GloVe-50). Then a Bayesian
comparison with Part 1's best. Every system runs on the same 6 splits as Part 1.

**Why compute the pooled vectors once?**
With frozen embeddings a bio's pooled vector never changes, so recomputing it every epoch is wasted work. Training
only the classifier on the stored vectors is the same model as the tutorial's frozen `EmbedAndPoolEncoder`, about
1 s per epoch instead of minutes.

**Why standardise the pooled vectors?**
Averaged fastText numbers are tiny and uneven in scale. Scaling each to mean 0 and spread 1 (training fold only) lets
one learning rate suit every pooling and size. It is the same as sklearn's StandardScaler in the §3 quick test.

**How do you test embedding size without changing the corpus?**
PCA shrinks the same fastText vectors to 50 or 100 numbers (the directions where words differ most). fastText-50 vs
GloVe-50 then compares training text at equal size: this fixes the §3 confound.

**How do you test whether pretraining helps?**
A "random start" system learns its word vectors from our bios only, with the same model and training. If fine-tuned
pretrained vectors beat it, pretraining added something.

**Fine-tuning trains 60.7 million numbers. How is that feasible on a laptop?**
Sparse updates: a batch only contains a few thousand distinct words, so only those rows of the table get a gradient
and an update (SparseAdam). The other ~200k rows are untouched in that step.

**Did you reuse the tutorial's training code?**
No. `experiment.train` picks the best epoch on the validation fold and then reports that same fold (optimistic).
We use Part 1's recipe: early stopping on an inner 10% hold-out of the training fold, so the validation fold is
used once.

## Q3 · Embed-and-pool (§4): results

**Which design choices mattered?** (all on 6 splits, Bayesian test, ROPE ±0.01)
- Pooling: mean beats max by 0.087 (0.746 vs 0.659, P = 1.00); adding max to mean: practically equal.
- Classifier: one hidden layer beats log-linear (+0.018, P = 0.97); a second layer adds nothing (P(equal) = 0.88).
- Embedding size (fastText shrunk with PCA): 50 → 0.668, 100 → 0.723, 300 → 0.746.
- Training text at equal size: fastText-50 0.668 vs GloVe-50 0.586 (+0.082). The §3 GloVe gap was half size, half corpus.
- Fine-tuning the vectors: 0.762 vs 0.746 frozen (+0.016, P(better) = 0.91). Best embed-and-pool system.

**Did pretrained embeddings add to your design?**
Yes: learning the vectors from scratch (same model) reaches 0.731, 0.031 below fine-tuned pretrained vectors (P = 1.00).
But versus Part 1's best (MLP + TF-IDF, 0.754) the gain is +0.008: inside the ROPE, P(practically equal) = 0.86. We beat
Part 1's log-linear + TF-IDF (0.749) with P = 0.97.

**Why isn't it clearly better than TF-IDF?**
Many bios contain a telling word (79% of dentist bios say "dentist"), which TF-IDF already uses. Embeddings add word
similarity, but pooling still ignores word order.

**Did you see over-fitting?**
Yes, with trained embeddings: training loss 0.18 vs 0.57 frozen by the last epoch, hold-out loss rising after epoch 2.
Fine-tuning peaks at epoch 2 (1.8 on average). Early stopping on the inner hold-out keeps the epoch-2 weights.

**Why did random-start embeddings do worse than frozen pretrained ones?**
60.7 million numbers learned from ~150k bios: rare words get few updates and the table memorises the training bios
(epoch 3 best). Pretrained vectors bring knowledge from 600 billion words of web text.

## Q4 · LSTM/GRU (§5): design questions (results pending)

**What does an LSTM add over embed-and-pool?**
Word order. It reads the bio word by word and updates a state (128 numbers) with learned gates that decide what to
keep, forget and add. Pooling treats "she is a nurse" and "a nurse is she" the same.

**LSTM vs GRU?**
Both are gated cells that avoid the vanishing-gradient problem of a plain RNN. The GRU has two gates instead of three
and no separate memory cell, so fewer weights; often equally good.

**Why test a mean readout?**
The tutorial classifies from the last state. After 60+ words the start of the bio may be faded, and bios often name the
occupation early. Averaging the states after every word lets every position count. After one epoch on a full split:
0.678 macro-F1 with mean readout vs 0.611 with the last state (a timing test, not a result).

**What is word dropout and why use it?**
While training, each word's vector is zeroed with some probability (we test 0.25). The model can't lean on any single
word, which reduces overfitting. Gal & Ghahramani (2016) show it is the right way to apply dropout in recurrent models;
the tutorial's `LSTMEncoder` implements it.

**How did you structure the search?**
One change at a time from the tutorial's LSTM (stage 1: 8 systems), then combine the changes that helped and vary the
embeddings and data size (stage 2). Same 6 splits, recipe and classifier as embed-and-pool, so only the encoder differs.

**Why not run it on the laptop?**
Measured: 26 s per epoch over 154k bios on the Mac GPU, about 0.4–0.8 h per system with frozen vectors and about 2 h
with trained vectors (6 splits). A teammate's NVIDIA GPU runs the grid with the same script; results come back as files.

**Does data size change the value of pretraining?**
Yes, for embed-and-pool (§4.4): fine-tuned fastText beats vectors learned from scratch by 0.094 with 10% of the training
data, 0.061 with 30% and 0.031 with all of it. Frozen fastText on 10% of the data still gets 0.702 (vs 0.746 with all);
from scratch drops to 0.615. Pretraining matters most when task data is scarce. The LSTM's curve is in stage 2.

# AI-assistance log

The course asks us to disclose AI use. One row per delegated task.

| Date | Tool | What was delegated | What we checked / changed |
|---|---|---|---|
| ≤ 2026-11-04 | AI assistant (earlier sessions) | Background guides (`docs/bias_in_bios_project1_guide.md`, `docs/project1_notebook_walkthrough.md`); draft skeleton; notebook §0–§2 | Isha: ⟨fill in what you verified⟩ |
| 2026-11-04 | Claude Code (Claude Opus 5.5) | Read brief, tutorial, `src/part1/`, docs; audited the notebook state; wrote `CLAUDE.md` | Isha: ⟨fill in⟩ |
| 2026-11-04 | Claude Code (Claude Opus 5.5) | Wrote `src/bib/` (style, tokenizer, encoders, models, evaluation, cv, tuning, data, plots), `scripts/run_grid.py`, `scripts/make_diagrams.py`; rebuilt the notebook §0–§8 (simpler text, new figures, diagrams, observations for §1–§3 and §6 from real outputs); ran §0–§3 and §6; tested every cell end to end on a 6k-bio subsample; wrote `docs/oral_prep.md` | Isha: ⟨fill in⟩ |
| 2026-11-04 | Claude Code (Claude Opus 5.5) | Switched to 3 folds × 2 seeds (Isha's request: 5 × 3 too slow), re-centred the tuning grid (C ∈ {0.3, 1, 3}); moved chart code to `bib/plots.py`; added plain-language "This cell" notes; wrote §9 (Bayesian, ROPE 0.01 + sensitivity), §10 (fairness), §11 (pronoun scrubbing via `bib/scrub.py`, log-linear pronoun weights), §12–§13; ran the whole notebook end to end (≈ 40 min) and wrote every observation from the outputs; replaced a `[4c]` example bio that contained a real name and phone number; rewrote `docs/oral_prep.md` | Isha: ⟨fill in⟩ |
| 2026-11-04 | Claude Code (Claude Opus 5.5) | Pilot experiments that changed decisions: full pool + 3 seeds (one fold < 1 min); per-system tuning of C / learning rate (TF-IDF at C = 1 was over-regularised); early stopping on macro-F1 instead of loss | Isha: ⟨fill in⟩ |

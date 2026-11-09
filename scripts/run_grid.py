"""Run the whole 3x3 cross-validation grid outside the notebook (leave it running).

    source .venv/bin/activate
    python scripts/run_grid.py                 # all 9 systems, cached ones are skipped
    python scripts/run_grid.py --archs loglin  # only one column of the grid
    python scripts/run_grid.py --rerun         # ignore caches

Needs (all produced by the notebook): results/week1/splits.json, the BPE tokenizer json and
results/week1/tuned_hparams.json (§5). Results go to results/week1/{arch}_{rep}.pkl, which §7/§8
of the notebook simply load.
"""
import argparse
import time
import warnings
from pathlib import Path

from tokenizers import Tokenizer

from bib import cv, data, encoders, models, tuning
from bib import tokenizer as btok

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "week1"
TOKENIZER = RESULTS / "bpe_lower_punct_v10000_mf5.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(encoders.REPRESENTATIONS))
    ap.add_argument("--archs", nargs="+", default=list(models.ARCHITECTURES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    texts, Y, G, folds = data.load_pool(RESULTS / "splits.json")
    tok = Tokenizer.from_file(str(TOKENIZER))
    C = encoders.count_matrix(btok.encode_all(tok, texts), tok.get_vocab_size())
    hp = tuning.load(RESULTS / "tuned_hparams.json")
    print(f"pool {len(Y):,} bios, seeds {list(folds)}, {len(folds[next(iter(folds))])} folds")

    t0 = time.time()
    for arch in args.archs:
        for rep in args.reps:
            cv.run_system(rep, arch, C, Y, folds, RESULTS, rerun=args.rerun, n_threads=args.threads,
                          hp=hp.get(cv.system_name(rep, arch)))
            print(f"--- {cv.system_name(rep, arch)} done, {(time.time() - t0) / 60:.1f} min elapsed")

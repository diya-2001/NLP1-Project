"""Run Part 2 systems over all 6 cross-validation splits, outside the notebook (laptop, GPU machine or Snellius).

    python scripts/run_part2.py --group rnn_stage1 --list        # only print the systems of a group
    python scripts/run_part2.py --group rnn_stage1 --quick       # 2-minute check: device + time estimate per system
    python scripts/run_part2.py --group rnn_stage1               # the real run (cached systems are skipped)
    python scripts/run_part2.py --group rnn_stage2 --best '{"cell": "gru", "readout": "mean"}'
    python scripts/run_part2.py --spec '{"kind": "rnn", "cell": "gru", "bidirectional": true}'

Groups (see bib.runs.group): pool_stage1, pool_stage2, pool_datasize (CPU, minutes); rnn_stage1, rnn_stage2 (GPU).
Each system is cached in results/part2/{name}.pkl, so a stopped run can simply be restarted. The notebook only loads
these files.

Needs these files (git-ignored; get them from the team bundle, see scripts/make_gpu_bundle.py):
results/part1/splits.json, results/part2/labels.npz, results/part2/emb_ft_crawl.npz, results/part2/ids_ft_crawl.npz.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import torch

from bib import neural, runs

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=runs.GROUPS, help="a planned group of systems")
    ap.add_argument("--best", default="{}", help="JSON settings of a stage-1 winner (for the *_stage2 groups)")
    ap.add_argument("--spec", action="append", default=[], help="JSON settings of one system (repeatable)")
    ap.add_argument("--seeds", nargs="+", type=int, help="only these seeds (default: all)")
    ap.add_argument("--device", help="force cpu / mps / cuda (default: automatic, CUDA when available)")
    ap.add_argument("--rerun", action="store_true", help="ignore cached results")
    ap.add_argument("--list", action="store_true", help="print the systems and stop")
    ap.add_argument("--quick", action="store_true",
                    help="no real run: 2 epochs per system on 10k bios of one split, then a time estimate")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    todo = []
    if args.group:
        todo += runs.group(args.group, json.loads(args.best))
    for s in args.spec:
        s = json.loads(s)
        todo.append(runs.system(s.pop("kind", "pool"), **s))
    print(f"{len(todo)} systems: {', '.join(n for n, _ in todo)}")
    if args.list:
        raise SystemExit

    runs.DEVICE = args.device
    print("CUDA available:", torch.cuda.is_available(),
          f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
    data = runs.Data.from_disk(ROOT)
    t0 = time.time()

    if args.quick:
        n_full = 0.9 * len(data.folds[0][0][0])                 # bios trained on per epoch in a real run
        for name, spec in todo:
            r = runs.run(name, {**spec, "max_epochs": 2}, data, limit=(10_000, 3_000))
            h = r["history"][(0, 0)]
            s_epoch = sum(x["seconds"] for x in h) / len(h) * n_full * spec.get("train_frac", 1) / 9_000
            print(f"{name:42s} on {neural.DEVICE}: {h[-1]['seconds']:.1f}s per epoch on 9k bios -> "
                  f"about {s_epoch * 8 * 6 / 3600:.1f} h for 6 splits (if ~8 epochs each)", flush=True)
        raise SystemExit

    for name, spec in todo:
        runs.run(name, spec, data, rerun=args.rerun, seeds=args.seeds)
        print(f"--- {name} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

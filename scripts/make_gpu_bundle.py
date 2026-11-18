"""Pack the git-ignored files a teammate needs to run Part 2 systems (e.g. the LSTM/GRU grid on a GPU).

    python scripts/make_gpu_bundle.py            # -> results/part2_gpu_bundle.zip (about 200 MB)

The teammate unzips it in the repository's top folder (the paths inside start with results/). It holds the
Part 1 splits, the labels, and the fastText vectors + word ids for our bios (made by notebook §0-§2 and [4a]).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["results/part1/splits.json", "results/part2/labels.npz",
         "results/part2/emb_ft_crawl.npz", "results/part2/ids_ft_crawl.npz"]

if __name__ == "__main__":
    out = ROOT / "results" / "part2_gpu_bundle.zip"
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run notebook part 2 up to [4a] first")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, arcname=f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

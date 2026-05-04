"""Build a side-by-side comparison grid across multiple sim runs.

Usage:
    python -m Simulation.compare_grid \\
        --row "ref:../data/ref_Tonkatsu2_3.0_2.5" \\
        --row "gt_rerender:results/gt_rerender_20260421/gt_30x25" \\
        --row "v1_stage2:results/rbcm_2setup_20260421/v1_stage2_3.0_2.5" \\
        --row "v2_stage2:results/rbcm_2setup_20260421/v2_stage2_3.0_2.5" \\
        --frames 01 02 03 04 05 06 07 08 \\
        --out figs/compare_30x25.png \\
        --title "Tonkatsu2 W=3.0 H=2.5"

Each --row is "<label>:<dir>" — the dir must contain config_XX.png files
(recursive search). If a label contains "diff" or a _cleandiff path, we
also show clean_snapdiff_XX.png from that dir.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def find_config(d: Path, idx: str) -> Path | None:
    """Find config_<idx>.png or clean_snapdiff_<idx>.png under d."""
    for name in (f"config_{idx}.png", f"clean_snapdiff_{idx}.png",
                 f"snapdiff_{idx}.png"):
        hits = list(d.rglob(name))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True,
                    help="<label>:<dir>, repeat for each row")
    ap.add_argument("--frames", nargs="+", default=["01","02","03","04",
                                                    "05","06","07","08"])
    ap.add_argument("--out",   required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi",   type=int, default=110)
    args = ap.parse_args()

    rows: List[Tuple[str, Path]] = []
    for r in args.row:
        if ":" not in r:
            raise ValueError(f"--row must be '<label>:<dir>' (got {r!r})")
        lab, p = r.split(":", 1)
        rows.append((lab, Path(p)))

    R, C = len(rows), len(args.frames)
    fig, ax = plt.subplots(R, C, figsize=(1.4 * C, 1.4 * R), dpi=args.dpi)
    if R == 1:
        ax = np.array([ax])
    if C == 1:
        ax = ax.reshape(-1, 1)

    for i, (lab, d) in enumerate(rows):
        for j, fidx in enumerate(args.frames):
            a = ax[i, j]
            p = find_config(d, fidx)
            if p is not None:
                img = Image.open(p).convert("RGB")
                a.imshow(np.array(img))
            else:
                a.set_facecolor("lightgray")
                a.text(0.5, 0.5, "missing",
                       ha="center", va="center", transform=a.transAxes,
                       fontsize=6, color="red")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(f"cfg_{fidx}", fontsize=8)
            if j == 0:
                a.set_ylabel(lab, fontsize=9, rotation=0,
                             ha="right", va="center", labelpad=40)

    if args.title:
        fig.suptitle(args.title, fontsize=11)

    plt.tight_layout(rect=[0.05, 0, 1, 0.96] if args.title else [0.05, 0, 1, 1])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"[compare_grid] wrote {out_path}")


if __name__ == "__main__":
    main()

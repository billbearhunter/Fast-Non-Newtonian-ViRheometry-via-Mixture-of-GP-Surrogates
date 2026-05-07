"""Generic snapdiff 6-row grid: Real vs MPM(θ̂) vs MPM(truth) + 3 pairwise diffs.

Layout per setup (6 rows × 9 frames):
    Row 1: Real            (camera y_obs)
    Row 2: MPM(θ̂)
    Row 3: MPM(truth)
    Row 4: |Real − MPM(θ̂)|       × amp
    Row 5: |Real − MPM(truth)|    × amp
    Row 6: |MPM(θ̂) − MPM(truth)| × amp

Two figures saved (setup1 + setup2 separate).

Generic — pass material name + ref dirs + render dirs + thetas.

Usage:
    python scripts/snapdiff_grid_v2.py --material Chuno \\
        --setup1-ref data/new_real_world_experiments/ref_Chuno_2.7_2.5_1 \\
        --setup2-ref data/new_real_world_experiments/ref_Chuno_2.0_4.0_2 \\
        --setup1-hat-render Simulation/results/snapdiff_chuno_y8q_new_setup1/0.66_8.12_19.65 \\
        --setup2-hat-render Simulation/results/snapdiff_chuno_y8q_new_setup2/0.66_8.12_19.65 \\
        --setup1-truth-render Simulation/results/snapdiff_truth_setup1/0.63_10.51_19.62 \\
        --setup2-truth-render Simulation/results/snapdiff_truth_setup2/0.63_10.51_19.62 \\
        --theta-hat 0.664 8.118 19.653 \\
        --theta-truth 0.633 10.51 19.62 \\
        --out-dir FlowCurve/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def load_grayscale(p: Path, size=(480, 270)) -> np.ndarray:
    return np.array(Image.open(p).resize(size).convert("L"))


def make_setup_grid(real_dir: Path, hat_dir: Path, truth_dir: Path,
                     fig_title: str, out_path: Path,
                     theta_hat: tuple[float, float, float],
                     theta_truth: tuple[float, float, float],
                     amp: float = 5.0):
    fig, axes = plt.subplots(6, 9, figsize=(20, 13))
    real_imgs = [load_grayscale(real_dir / f"config_{f:02d}.png") for f in range(9)]
    hat_imgs  = [load_grayscale(hat_dir  / f"config_{f:02d}.png") for f in range(9)]
    tr_imgs   = [load_grayscale(truth_dir / f"config_{f:02d}.png") for f in range(9)]

    for i, f in enumerate(range(9)):
        axes[0, i].imshow(real_imgs[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off"); axes[0, i].set_title(f"f{f}", fontsize=10)
        axes[1, i].imshow(hat_imgs[i], cmap="gray", vmin=0, vmax=255); axes[1, i].axis("off")
        axes[2, i].imshow(tr_imgs[i], cmap="gray", vmin=0, vmax=255);  axes[2, i].axis("off")
        d4 = np.abs(real_imgs[i].astype(int) - hat_imgs[i].astype(int))
        axes[3, i].imshow(np.clip(d4 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255); axes[3, i].axis("off")
        d5 = np.abs(real_imgs[i].astype(int) - tr_imgs[i].astype(int))
        axes[4, i].imshow(np.clip(d5 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255); axes[4, i].axis("off")
        d6 = np.abs(hat_imgs[i].astype(int) - tr_imgs[i].astype(int))
        axes[5, i].imshow(np.clip(d6 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255); axes[5, i].axis("off")

    row_labels = ["Real", "MPM(θ̂)", "MPM(truth)",
                   f"|Real − MPM(θ̂)|×{amp:g}",
                   f"|Real − MPM(truth)|×{amp:g}",
                   f"|MPM(θ̂) − MPM(truth)|×{amp:g}"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].text(-90, 135, lbl, fontsize=11, rotation=90, va="center", ha="center")

    th_str = f"θ̂=({theta_hat[0]:.3f}, {theta_hat[1]:.2f}, {theta_hat[2]:.2f})"
    tr_str = f"truth=({theta_truth[0]:.3f}, {theta_truth[1]:.2f}, {theta_truth[2]:.2f})"
    fig.suptitle(f"{fig_title}\n{th_str}    {tr_str}", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.02, wspace=0.04, hspace=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--material", required=True)
    ap.add_argument("--setup1-ref", type=Path, required=True)
    ap.add_argument("--setup2-ref", type=Path, required=True)
    ap.add_argument("--setup1-hat-render", type=Path, required=True)
    ap.add_argument("--setup2-hat-render", type=Path, required=True)
    ap.add_argument("--setup1-truth-render", type=Path, required=True)
    ap.add_argument("--setup2-truth-render", type=Path, required=True)
    ap.add_argument("--theta-hat", nargs=3, type=float, required=True,
                    metavar=("n", "eta", "sigma_y"))
    ap.add_argument("--theta-truth", nargs=3, type=float, required=True,
                    metavar=("n", "eta", "sigma_y"))
    ap.add_argument("--setup1-name", default="setup1",
                    help="Display label for setup1 (e.g. 'setup1 (W=2.5, H=2.7)')")
    ap.add_argument("--setup2-name", default="setup2")
    ap.add_argument("--out-dir", type=Path, default=REPO / "FlowCurve" / "figs")
    ap.add_argument("--amp", type=float, default=5.0)
    args = ap.parse_args()

    make_setup_grid(
        real_dir =args.setup1_ref,
        hat_dir  =args.setup1_hat_render,
        truth_dir=args.setup1_truth_render,
        fig_title=f"{args.material} {args.setup1_name} — Real vs MPM(θ̂_y8q) vs MPM(truth)",
        out_path =args.out_dir / f"snapdiff_{args.material.lower()}_setup1_v2.png",
        theta_hat=tuple(args.theta_hat),
        theta_truth=tuple(args.theta_truth),
        amp=args.amp,
    )
    make_setup_grid(
        real_dir =args.setup2_ref,
        hat_dir  =args.setup2_hat_render,
        truth_dir=args.setup2_truth_render,
        fig_title=f"{args.material} {args.setup2_name} — Real vs MPM(θ̂_y8q) vs MPM(truth)",
        out_path =args.out_dir / f"snapdiff_{args.material.lower()}_setup2_v2.png",
        theta_hat=tuple(args.theta_hat),
        theta_truth=tuple(args.theta_truth),
        amp=args.amp,
    )


if __name__ == "__main__":
    main()

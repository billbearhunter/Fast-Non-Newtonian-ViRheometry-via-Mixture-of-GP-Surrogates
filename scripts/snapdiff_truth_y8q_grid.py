"""Snapdiff grid: Real vs MPM(θ̂_y8q) vs MPM(truth), with all 3 pairwise diffs.

Layout per setup (6 rows × 9 frames):
    Row 1:  Real            (camera y_obs)
    Row 2:  MPM(θ̂_y8q)
    Row 3:  MPM(truth)
    Row 4:  |Real - MPM(θ̂_y8q)| × 5      (sim-real gap of inverse θ̂)
    Row 5:  |Real - MPM(truth)|  × 5      (pure sim-real gap, surface tension etc.)
    Row 6:  |MPM(θ̂_y8q) - MPM(truth)| × 5  (inverse error in sim-only space)

Two figures saved (setup1 + setup2 separate) per user request.

Reference layout: FlowCurve/figs/snapdiff_y8w5_k1_f1w15_grid.png (3-row).
This is the 6-row extension with truth comparison.

Usage:
    python scripts/snapdiff_truth_y8q_grid.py
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
    """Render 6-row × 9-col grid for one setup."""
    fig, axes = plt.subplots(6, 9, figsize=(20, 13))

    real_imgs = [load_grayscale(real_dir / f"config_{f:02d}.png") for f in range(9)]
    hat_imgs  = [load_grayscale(hat_dir  / f"config_{f:02d}.png") for f in range(9)]
    tr_imgs   = [load_grayscale(truth_dir / f"config_{f:02d}.png") for f in range(9)]

    for i, f in enumerate(range(9)):
        # Row 1: Real
        axes[0, i].imshow(real_imgs[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"f{f}", fontsize=10)
        # Row 2: MPM(θ̂)
        axes[1, i].imshow(hat_imgs[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
        # Row 3: MPM(truth)
        axes[2, i].imshow(tr_imgs[i], cmap="gray", vmin=0, vmax=255)
        axes[2, i].axis("off")
        # Row 4: |Real - MPM(θ̂)| × amp
        d4 = np.abs(real_imgs[i].astype(int) - hat_imgs[i].astype(int))
        axes[3, i].imshow(np.clip(d4 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255)
        axes[3, i].axis("off")
        # Row 5: |Real - MPM(truth)| × amp
        d5 = np.abs(real_imgs[i].astype(int) - tr_imgs[i].astype(int))
        axes[4, i].imshow(np.clip(d5 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255)
        axes[4, i].axis("off")
        # Row 6: |MPM(θ̂) - MPM(truth)| × amp
        d6 = np.abs(hat_imgs[i].astype(int) - tr_imgs[i].astype(int))
        axes[5, i].imshow(np.clip(d6 * amp, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255)
        axes[5, i].axis("off")

    # Row labels (left margin)
    row_labels = ["Real", f"MPM(θ̂)\nθ̂_y8q", "MPM(truth)\nθ_truth",
                   f"|Real − MPM(θ̂)|×{amp:g}", f"|Real − MPM(truth)|×{amp:g}",
                   f"|MPM(θ̂) − MPM(truth)|×{amp:g}"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].text(-90, 135, lbl, fontsize=11, rotation=90, va="center", ha="center")

    th_str = f"θ̂_y8q=({theta_hat[0]:.3f}, {theta_hat[1]:.2f}, {theta_hat[2]:.2f})"
    tr_str = f"θ_truth=({theta_truth[0]:.3f}, {theta_truth[1]:.2f}, {theta_truth[2]:.2f})"
    fig.suptitle(f"{fig_title}\n{th_str}    {tr_str}", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.02, wspace=0.04, hspace=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--theta-hat", nargs=3, type=float,
                    default=[0.6985, 7.911, 20.867],
                    metavar=("n", "eta", "sigma_y"),
                    help="Recovered θ̂ (matches existing y8q render dir)")
    ap.add_argument("--theta-truth", nargs=3, type=float,
                    default=[0.633, 10.51, 19.62],
                    metavar=("n", "eta", "sigma_y"),
                    help="Rheometer truth")
    ap.add_argument("--hat-tag", default="0.70_7.91_20.87",
                    help="θ̂ render dir name (under Simulation/results/snapdiff_joint_y8q_setup{1,2}/)")
    ap.add_argument("--truth-tag", default="0.63_10.51_19.62",
                    help="truth render dir name (under Simulation/results/snapdiff_truth_setup{1,2}/)")
    args = ap.parse_args()

    # Setup1
    make_setup_grid(
        real_dir=REPO / "data" / "new_real_world_experiments" / "ref_Chuno_2.7_2.5_1",
        hat_dir =REPO / "Simulation" / "results" / "snapdiff_joint_y8q_setup1" / args.hat_tag,
        truth_dir=REPO / "Simulation" / "results" / "snapdiff_truth_setup1" / args.truth_tag,
        fig_title="Chuno setup1  (W=2.5, H=2.7)  —  Real vs MPM(θ̂_y8q) vs MPM(truth)",
        out_path =REPO / "FlowCurve" / "figs" / "snapdiff_truth_y8q_setup1.png",
        theta_hat=tuple(args.theta_hat),
        theta_truth=tuple(args.theta_truth),
    )

    # Setup2
    make_setup_grid(
        real_dir=REPO / "data" / "new_real_world_experiments" / "ref_Chuno_2.0_4.0_2",
        hat_dir =REPO / "Simulation" / "results" / "snapdiff_joint_y8q_setup2" / args.hat_tag,
        truth_dir=REPO / "Simulation" / "results" / "snapdiff_truth_setup2" / args.truth_tag,
        fig_title="Chuno setup2  (W=4.0, H=2.0)  —  Real vs MPM(θ̂_y8q) vs MPM(truth)",
        out_path =REPO / "FlowCurve" / "figs" / "snapdiff_truth_y8q_setup2.png",
        theta_hat=tuple(args.theta_hat),
        theta_truth=tuple(args.theta_truth),
    )


if __name__ == "__main__":
    main()

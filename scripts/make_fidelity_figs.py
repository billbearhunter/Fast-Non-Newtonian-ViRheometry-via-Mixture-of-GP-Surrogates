"""
Render §7.2 figures from the oracle-routing fidelity outputs.

Inputs
------
  data/eval_oracle/parity_data_oracle.npz
    keys: y_true (N,8), y_pred (N,8), sigma_pred (N,8), sub_ids (N,), gids (N,)
  data/eval_oracle/expert_r2_oracle.csv
    columns: gid, sub_id, n_samples, R2_overall, MAE_overall_cm

Outputs (written into the sibling paper repo's figs/ folder by default)
-----------------------------------------------------------------------
  figs/true_vs_pred.png    — fig:parity
  figs/moe_r2_boxplot.png  — fig:expert-r2
  figs/sigma_calibration.png — optional supplementary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / "Real-Time-Non-Newtonian-ViRheometry-via-Surrogate-Accelerated-Optimization"


def parity_figure(npz: Path, out: Path, sample: int = 8000, seed: int = 0):
    d = np.load(npz)
    y_true = d["y_true"]
    y_pred = d["y_pred"]
    N, D = y_true.shape

    rng = np.random.default_rng(seed)
    if N > sample:
        idx = rng.choice(N, size=sample, replace=False)
    else:
        idx = np.arange(N)
    yt = y_true[idx]
    yp = y_pred[idx]

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6),
                             sharex=False, sharey=False)
    axes = axes.flatten()
    for d_idx in range(D):
        ax = axes[d_idx]
        x = yt[:, d_idx]
        y = yp[:, d_idx]
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.scatter(x, y, s=2.0, alpha=0.30, color="#1f77b4",
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # full-data R² for this dim
        yt_full = y_true[:, d_idx]; yp_full = y_pred[:, d_idx]
        ss_res = float(np.sum((yt_full - yp_full) ** 2))
        ss_tot = float(np.sum((yt_full - yt_full.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        mae = float(np.mean(np.abs(yt_full - yp_full)))
        ax.set_title(f"frame {d_idx+1}: $R^2$={r2:.4f}, MAE={mae:.3f} cm",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        if d_idx % 4 == 0:
            ax.set_ylabel(r"$\hat y_d$ (cm)", fontsize=9)
        if d_idx >= 4:
            ax.set_xlabel(r"$y_d$ (cm)", fontsize=9)
    fig.suptitle(
        f"Surrogate parity on $T_{{\\mathrm{{sim}}}}$ "
        f"($N={N}$ held-out MPM rows, oracle routing); "
        f"shown: random {len(idx)}-row subsample.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[parity] wrote {out}")


def per_sub_r2_figure(csv: Path, out: Path):
    df = pd.read_csv(csv)
    gids = sorted(df["gid"].unique().tolist())
    data = [df.loc[df["gid"] == g, "R2_overall"].to_numpy() for g in gids]

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    bp = ax.boxplot(data, positions=gids, widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", markersize=2,
                                    markerfacecolor="#888", markeredgecolor="none"),
                    medianprops=dict(color="#d62728", lw=1.2),
                    boxprops=dict(color="#1f77b4", lw=0.9),
                    whiskerprops=dict(color="#1f77b4", lw=0.9),
                    capprops=dict(color="#1f77b4", lw=0.9))
    med_all = float(df["R2_overall"].median())
    p5_all = float(df["R2_overall"].quantile(0.05))
    ax.axhline(med_all, color="#d62728", lw=0.6, ls="--", alpha=0.7,
               label=f"global median $R^2$ = {med_all:.3f}")
    ax.axhline(p5_all, color="#888", lw=0.6, ls=":", alpha=0.7,
               label=f"global $p_5$ $R^2$ = {p5_all:.3f}")
    ax.set_xlabel("geometry sub-bank index $g$", fontsize=10)
    ax.set_ylabel(r"per-sub $R^2$", fontsize=10)
    ax.set_title(f"Per-sub-expert $R^2$ across all "
                 f"{len(df)} sub-experts (oracle routing)", fontsize=10)
    ax.set_ylim(0.80, 1.001)
    ax.set_xticks(gids)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[per-sub-R2] wrote {out}  "
          f"(median={med_all:.4f}, p5={p5_all:.4f}, "
          f"min={df['R2_overall'].min():.4f})")


def sigma_calibration_figure(npz: Path, out: Path):
    d = np.load(npz)
    err = (d["y_pred"] - d["y_true"]).ravel()
    sig = d["sigma_pred"].ravel()
    z = err / np.maximum(sig, 1e-12)
    # bin by predicted σ
    bins = np.quantile(sig, np.linspace(0, 1, 21))
    bins[0] = bins[0] - 1e-9
    bins[-1] = bins[-1] + 1e-9
    bin_idx = np.digitize(sig, bins) - 1
    centers = []
    rmse_emp = []
    sig_mean = []
    for b in range(20):
        m = bin_idx == b
        if m.sum() < 30:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        rmse_emp.append(np.sqrt(np.mean(err[m] ** 2)))
        sig_mean.append(np.mean(sig[m]))
    centers = np.array(centers); rmse_emp = np.array(rmse_emp); sig_mean = np.array(sig_mean)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Left: σ_pred vs empirical RMSE
    lo = float(min(sig_mean.min(), rmse_emp.min()))
    hi = float(max(sig_mean.max(), rmse_emp.max()))
    axes[0].scatter(sig_mean, rmse_emp, s=18, color="#1f77b4")
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].set_xlabel(r"binned mean $\sigma_{\mathrm{pred}}$ (cm)", fontsize=9)
    axes[0].set_ylabel("empirical RMSE in bin (cm)", fontsize=9)
    axes[0].set_title("Calibration of GP posterior $\\sigma$", fontsize=10)
    axes[0].grid(True, alpha=0.3, lw=0.5)
    axes[0].tick_params(labelsize=8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)

    # Right: histogram of standardized residuals z = err/σ
    z_clip = z[np.abs(z) < 10]
    axes[1].hist(z_clip, bins=80, density=True, color="#1f77b4",
                 edgecolor="none", alpha=0.85, label="empirical $z$")
    xx = np.linspace(-6, 6, 400)
    axes[1].plot(xx, np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi),
                 "k-", lw=1.0, label=r"$\mathcal{N}(0,1)$")
    axes[1].set_xlim(-6, 6)
    axes[1].set_xlabel(r"standardized residual $z = (\hat y - y)/\sigma_{\mathrm{pred}}$",
                       fontsize=9)
    axes[1].set_ylabel("density", fontsize=9)
    axes[1].set_title(f"Residual distribution (std={z.std():.2f})", fontsize=10)
    axes[1].grid(True, alpha=0.3, lw=0.5)
    axes[1].tick_params(labelsize=8)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[sigma-cal] wrote {out}  "
          f"(z mean={z.mean():.3f}, std={z.std():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    default=REPO / "data" / "eval_oracle" / "parity_data_oracle.npz")
    ap.add_argument("--csv", type=Path,
                    default=REPO / "data" / "eval_oracle" / "expert_r2_oracle.csv")
    ap.add_argument("--figs-dir", type=Path,
                    default=PAPER / "figs")
    args = ap.parse_args()

    parity_figure(args.npz, args.figs_dir / "true_vs_pred.png")
    per_sub_r2_figure(args.csv, args.figs_dir / "moe_r2_boxplot.png")
    sigma_calibration_figure(args.npz, args.figs_dir / "sigma_calibration.png")


if __name__ == "__main__":
    main()

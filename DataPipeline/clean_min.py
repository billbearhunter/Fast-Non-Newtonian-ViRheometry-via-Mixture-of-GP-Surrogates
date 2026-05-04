"""Minimum cleaning pass on the labeled training set.

Applies four filters identified in the 2026-04-18 data audit as
definite non-physical / splash-filter-leak rows:

  1. Re_dam  > 100        (inertia-dominated: splash filter leak from CCR)
  2. eta     < 0.01       (water-like; HB degenerates)
  3. sigma_y < 0.01       (no yield stress; HB degenerates to Newtonian)
  4. x_08 .. x_01 not monotonically non-decreasing  (simulation artefact)

Writes:
  <out_ws>/train_labeled.csv
  <out_ws>/val_labeled.csv
  <out_ws>/test_labeled.csv
  <out_ws>/dropped_rows.csv        (every dropped row with reason tag)
  <out_ws>/clean_report.txt        (summary)

Usage:
  python -m DataPipeline.clean_min                 # defaults below
  python -m DataPipeline.clean_min --aggressive    # Re>50, eta<0.1, sy<0.1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DEFAULT = ROOT / "Optimization" / "moe_workspace_rebuild_kphi20"
DST_DEFAULT = ROOT / "Optimization" / "moe_workspace_rebuild_kphi20_clean"

X_COLS = [f"x_{i:02d}" for i in range(1, 9)]
SPLITS = ("train", "val", "test")

RHO = 1.0       # g/cm^3
G   = 981.0     # cm/s^2


def compute_re(df):
    return RHO * df["height"].values ** 1.5 * np.sqrt(G) / np.maximum(df["eta"].values, 1e-12)


def tag_dropped(df: pd.DataFrame,
                re_thresh: float, eta_thresh: float, sy_thresh: float,
                mono_eps: float) -> pd.DataFrame:
    """Return a copy of df with a 'drop_reason' column (empty string = keep)."""
    df = df.copy()
    Re = compute_re(df)

    reason = np.array([""] * len(df), dtype=object)
    # reasons stack — first match wins for legibility but we annotate all
    splash = Re > re_thresh
    low_eta = df["eta"].values < eta_thresh
    low_sy  = df["sigma_y"].values < sy_thresh
    X = df[X_COLS].values
    dX = np.diff(X, axis=1)
    nonmono = (dX < -mono_eps).any(axis=1)

    for mask, tag in [(splash, "splash_Re"),
                      (low_eta, "low_eta"),
                      (low_sy, "low_sigma_y"),
                      (nonmono, "nonmonotonic")]:
        has = mask & (reason == "")
        reason[has] = tag
        # annotate multi-flag
        multi = mask & (reason != "") & ~has
        for i in np.where(multi)[0]:
            if tag not in reason[i]:
                reason[i] = reason[i] + f"+{tag}"

    df["drop_reason"] = reason
    df["Re_dam"] = Re
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--dst", default=str(DST_DEFAULT))
    ap.add_argument("--re",     type=float, default=100.0,
                    help="Drop rows with Re_dam > this (default 100 = splash filter)")
    ap.add_argument("--eta",    type=float, default=0.01,
                    help="Drop rows with eta < this (default 0.01)")
    ap.add_argument("--sy",     type=float, default=0.01,
                    help="Drop rows with sigma_y < this (default 0.01)")
    ap.add_argument("--mono-eps", type=float, default=0.02,
                    help="Allow this much non-monotonicity (cm) before dropping")
    ap.add_argument("--aggressive", action="store_true",
                    help="Shortcut: Re=50, eta=0.1, sy=0.1")
    args = ap.parse_args()

    if args.aggressive:
        args.re, args.eta, args.sy = 50.0, 0.1, 0.1

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"source : {src}")
    print(f"dest   : {dst}")
    print(f"filter : Re>{args.re}  eta<{args.eta}  sigma_y<{args.sy}  mono_eps={args.mono_eps}")

    rows_in   = {}
    rows_out  = {}
    all_dropped = []
    report_lines = []

    report_lines.append(f"# clean_min.py report")
    report_lines.append(f"# filter: Re>{args.re}  eta<{args.eta}  "
                        f"sigma_y<{args.sy}  mono_eps={args.mono_eps}\n")

    for split in SPLITS:
        src_csv = src / f"{split}_labeled.csv"
        if not src_csv.exists():
            print(f"  skip {split}: {src_csv} not found")
            continue
        df = pd.read_csv(src_csv)
        rows_in[split] = len(df)

        tagged = tag_dropped(df, args.re, args.eta, args.sy, args.mono_eps)
        keep = tagged[tagged["drop_reason"] == ""].drop(
            columns=["drop_reason", "Re_dam"])
        drop = tagged[tagged["drop_reason"] != ""].copy()
        drop["_split"] = split
        all_dropped.append(drop)

        dst_csv = dst / f"{split}_labeled.csv"
        keep.to_csv(dst_csv, index=False)
        rows_out[split] = len(keep)

        # per-reason counts
        reason_counts = drop["drop_reason"].value_counts().to_dict()
        print(f"  {split}: {len(df):,} → {len(keep):,} "
              f"(dropped {len(df)-len(keep):,}, {(len(df)-len(keep))/len(df)*100:.2f}%)")
        for r, c in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"      {r}: {c}")
        report_lines.append(f"{split}: in={len(df):,} out={len(keep):,} "
                            f"dropped={len(df)-len(keep):,}")
        for r, c in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            report_lines.append(f"  {r}: {c}")
        report_lines.append("")

    if all_dropped:
        dropped_df = pd.concat(all_dropped, ignore_index=True)
        dropped_df.to_csv(dst / "dropped_rows.csv", index=False)
        print(f"  saved dropped rows → {dst/'dropped_rows.csv'} "
              f"({len(dropped_df):,} rows)")

    total_in  = sum(rows_in.values())
    total_out = sum(rows_out.values())
    summary = (f"\nTOTAL: in={total_in:,}  out={total_out:,}  "
               f"dropped={total_in-total_out:,} "
               f"({(total_in-total_out)/max(1,total_in)*100:.3f}%)")
    print(summary)
    report_lines.append(summary)
    (dst / "clean_report.txt").write_text("\n".join(report_lines))
    print(f"  report → {dst/'clean_report.txt'}")


if __name__ == "__main__":
    main()

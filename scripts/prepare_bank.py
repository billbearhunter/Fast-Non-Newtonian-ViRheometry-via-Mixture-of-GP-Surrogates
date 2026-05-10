"""Prepare bank for training: dedup + add train/val split column.

1. Dedup check: report exact duplicate (n,eta,sigma_y,w,h) rows across all
   sub_assignments.csv files in the bank.
2. Add 'split' column (train/val) to every sub_assignments.csv.
   Strategy: within each sub, 15% of rows → val (random, seeded by sub_id).
   Min val size: 30 rows (skip val for tiny subs).

Usage:
    python scripts/prepare_bank.py --bank Models/v10_yshape_round2_sy
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

KEY_COLS  = ["n", "eta", "sigma_y", "width", "height"]
VAL_FRAC  = 0.15
MIN_VAL_N = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=REPO / "Models" / "v10_yshape_round2_sy")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--min-val-n", type=int, default=MIN_VAL_N)
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()

    csv_paths = sorted(args.bank.glob("state_gid_*/sub_assignments.csv"))
    print(f"Bank: {args.bank}")
    print(f"Found {len(csv_paths)} sub_assignments.csv files")

    # ── 1. Dedup check ───────────────────────────────────────────────────────
    if not args.skip_dedup:
        print("\n=== Dedup check (exact match on n,eta,sigma_y,w,h) ===")
        all_keys = []
        for p in csv_paths:
            df = pd.read_csv(p, usecols=KEY_COLS)
            all_keys.append(df)
        combined = pd.concat(all_keys, ignore_index=True)
        total = len(combined)
        dupes = combined.duplicated(subset=KEY_COLS).sum()
        print(f"  Total rows: {total:,}")
        print(f"  Exact duplicates: {dupes:,}  ({dupes/total*100:.2f}%)")
        if dupes > 0:
            dup_rows = combined[combined.duplicated(subset=KEY_COLS, keep=False)]
            print(f"  Sample duplicate params:")
            print(dup_rows.drop_duplicates(subset=KEY_COLS).head(5).to_string(index=False))
        else:
            print("  No duplicates found ✓")

    # ── 2. Add train/val split column ────────────────────────────────────────
    print("\n=== Adding train/val split column ===")
    total_train = total_val = total_skip = 0

    for p in csv_paths:
        df = pd.read_csv(p)
        if "split" in df.columns:
            df = df.drop(columns=["split"])

        split_col = np.full(len(df), "train", dtype=object)

        for sid in df["sub_id"].unique():
            mask = df["sub_id"] == sid
            idx  = np.where(mask)[0]
            n_sub = len(idx)
            n_val = max(int(round(n_sub * args.val_frac)), 0)

            if n_val < args.min_val_n:
                total_skip += n_sub
                continue  # too small: all train

            rng = np.random.default_rng(int(sid) * 97 + 13)
            val_local = rng.choice(n_sub, n_val, replace=False)
            val_global = idx[val_local]
            split_col[val_global] = "val"

            total_train += n_sub - n_val
            total_val   += n_val

        df["split"] = split_col
        df.to_csv(p, index=False)

    print(f"  train rows: {total_train:,}")
    print(f"  val   rows: {total_val:,}  ({total_val/(total_train+total_val)*100:.1f}%)")
    print(f"  skipped (sub too small): {total_skip:,} rows → all train")
    print(f"\nDone. All sub_assignments.csv updated with 'split' column.")


if __name__ == "__main__":
    main()

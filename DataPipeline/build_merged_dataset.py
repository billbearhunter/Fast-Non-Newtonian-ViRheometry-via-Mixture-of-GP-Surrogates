"""Build a single merged dataset from all current labelled sources.

Does NOT modify any source file — writes to a fresh output directory.

Sources
-------
  1. Current training corpus (verbatim):
       moe_workspace_rebuild_kphi20_clean/{train,val,test}_labeled.csv
  2. Batch-1 BO-infill (completed):
       moe_workspace_bo_infill_20260419/infill_clean.csv
  3. Old BO scan (34 files, K-sample consistency-checked):
       moe_workspace_bo_infill_20260419/old_bo_scan/old_bo_usable.csv
  4. Batch-2 BO-infill (optional; pass --include-batch2 after it finishes):
       moe_workspace_bo_infill_20260419_batch2/infill_clean.csv

Hygiene rules
-------------
  * val + test are copied VERBATIM (so metrics remain comparable with the
    already-trained hierarchical MoE).
  * All NEW sources (batch-1 / oldbo / batch-2) are deduplicated against
    val ∪ test ∪ existing-train at 4-dp input precision before being
    appended to train.
  * Cross-source duplicates are also removed as sources are accepted in
    order (batch1 → oldbo → batch2).
  * Every row carries a `_src` tag (train / val / test / batch1 / oldbo / batch2).
  * A final self-dedup pass on the merged train is a cheap sanity check.

Output
------
  <out_dir>/train_merged.csv
  <out_dir>/val_merged.csv
  <out_dir>/test_merged.csv
  <out_dir>/merge_report.json

Usage
-----
  python -m DataPipeline.build_merged_dataset
  python -m DataPipeline.build_merged_dataset --include-batch2
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()

CORE_IN  = ["n", "eta", "sigma_y", "width", "height"]
CORE_OUT = [f"x_{i:02d}" for i in range(1, 9)]
CORE = CORE_IN + CORE_OUT
DEDUP_PREC = 4   # decimal places for duplicate detection


def _load(p: Path, src_tag: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} missing columns: {missing}")
    keep = CORE + [c for c in ("cluster_id", "cluster_conf") if c in df.columns]
    df = df[keep].copy()
    df["_src"] = src_tag
    return df


def _dedup_against(new_df: pd.DataFrame, ref_keys: set) -> tuple[pd.DataFrame, int]:
    keys = list(map(tuple, new_df[CORE_IN].round(DEDUP_PREC).to_numpy()))
    mask = np.array([k not in ref_keys for k in keys], dtype=bool)
    return new_df.loc[mask].reset_index(drop=True), int((~mask).sum())


def _within_dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys_df = pd.DataFrame(df[CORE_IN].round(DEDUP_PREC).to_numpy(), columns=CORE_IN)
    dup = keys_df.duplicated(keep="first")
    return df.loc[~dup].reset_index(drop=True), int(dup.sum())


def _strict_monotonic_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where x_01 <= x_02 <= ... <= x_08 is violated (strict)."""
    mask = pd.Series(True, index=df.index)
    for i in range(len(CORE_OUT) - 1):
        mask &= df[CORE_OUT[i + 1]] >= df[CORE_OUT[i]]
    return df.loc[mask].reset_index(drop=True), int((~mask).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(
        ROOT / "Optimization/moe_workspace_rebuild_kphi20_clean"))
    ap.add_argument("--batch1-csv", default=str(
        ROOT / "Optimization/moe_workspace_bo_infill_20260419/infill_clean.csv"))
    ap.add_argument("--oldbo-csv", default=str(
        ROOT / "Optimization/moe_workspace_bo_infill_20260419/old_bo_scan/old_bo_usable.csv"))
    ap.add_argument("--batch2-csv", default=str(
        ROOT / "Optimization/moe_workspace_bo_infill_20260419_batch2/infill_clean.csv"))
    ap.add_argument("--include-batch2", action="store_true", default=False)
    ap.add_argument("--strict-monotonic", action="store_true", default=False,
                    help="Drop rows where x_01..x_08 is not strictly non-decreasing.")
    ap.add_argument("--out-dir", default=str(
        ROOT / "Optimization/moe_workspace_merged_v1_20260419"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] out_dir: {out_dir}")

    # ── Existing splits ─────────────────────────────────────────────────
    train_dir = Path(args.train_dir)
    df_train = _load(train_dir / "train_labeled.csv", "train")
    df_val   = _load(train_dir / "val_labeled.csv",   "val")
    df_test  = _load(train_dir / "test_labeled.csv",  "test")
    print(f"[merge] existing: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # ── Reference key-set: val ∪ test ∪ existing-train ──────────────────
    full_ref = set()
    for ref in (df_val, df_test, df_train):
        full_ref.update(map(tuple, ref[CORE_IN].round(DEDUP_PREC).to_numpy()))
    print(f"[merge] reference keyset size: {len(full_ref):,}")

    # ── New sources ──────────────────────────────────────────────────────
    stats = {}
    new_pieces = []
    for tag, path, enabled in [
        ("batch1", Path(args.batch1_csv), True),
        ("oldbo",  Path(args.oldbo_csv),  True),
        ("batch2", Path(args.batch2_csv), args.include_batch2),
    ]:
        if not enabled:
            print(f"[merge] {tag}: skipped (flag off)")
            stats[tag] = {"status": "skipped_flag"}
            continue
        if not path.exists():
            print(f"[merge] {tag}: SKIP, not found -> {path}")
            stats[tag] = {"status": "missing", "path": str(path)}
            continue
        df = _load(path, tag)
        n_before = len(df)
        df, n_dup = _dedup_against(df, full_ref)
        full_ref.update(map(tuple, df[CORE_IN].round(DEDUP_PREC).to_numpy()))
        new_pieces.append(df)
        stats[tag] = {"raw": n_before, "after_ref_dedup": len(df),
                      "dropped_ref_dup": n_dup, "path": str(path)}
        print(f"[merge] {tag}: {n_before:,} -> {len(df):,}  (dropped {n_dup:,} ref-dup)")

    # ── Merged train ─────────────────────────────────────────────────────
    merged_train = pd.concat([df_train] + new_pieces, ignore_index=True)
    merged_train, n_self = _within_dedup(merged_train)
    print(f"[merge] final train: {len(merged_train):,}  (final self-dedup removed {n_self})")

    # ── Optional strict monotonicity filter ───────────────────────────────
    mono_stats = {"applied": False}
    if args.strict_monotonic:
        per_src_before = merged_train["_src"].value_counts().to_dict()
        merged_train, n_mono = _strict_monotonic_filter(merged_train)
        per_src_after = merged_train["_src"].value_counts().to_dict()
        per_src_drop = {k: per_src_before.get(k, 0) - per_src_after.get(k, 0)
                        for k in per_src_before}
        mono_stats = {"applied": True,
                      "total_dropped": n_mono,
                      "per_src_dropped": per_src_drop}
        print(f"[merge] strict-monotonic: dropped {n_mono:,}  "
              f"(per-src: {per_src_drop})")
        # Also scrub val / test even though they had 0 earlier — be explicit.
        df_val,  n_v = _strict_monotonic_filter(df_val)
        df_test, n_t = _strict_monotonic_filter(df_test)
        if n_v or n_t:
            print(f"[merge] strict-monotonic: val dropped {n_v}  test dropped {n_t}")

    # ── Write ────────────────────────────────────────────────────────────
    merged_train.to_csv(out_dir / "train_merged.csv", index=False)
    df_val.to_csv(out_dir / "val_merged.csv", index=False)
    df_test.to_csv(out_dir / "test_merged.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "sources": {
            "train_src": {"rows": len(df_train), "dir": str(train_dir)},
            "val_src":   {"rows": len(df_val),   "dir": str(train_dir)},
            "test_src":  {"rows": len(df_test),  "dir": str(train_dir)},
            **stats,
        },
        "merged": {
            "train_merged_rows": len(merged_train),
            "val_merged_rows":   len(df_val),
            "test_merged_rows":  len(df_test),
            "total_rows": len(merged_train) + len(df_val) + len(df_test),
        },
        "per_src_in_train_merged":
            merged_train["_src"].value_counts().to_dict(),
        "dedup_precision_decimals": DEDUP_PREC,
        "final_self_dedup_removed": n_self,
        "strict_monotonic": mono_stats,
    }
    with open(out_dir / "merge_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n[merge] DONE")
    print(f"  train_merged: {len(merged_train):,}")
    print(f"  val_merged:   {len(df_val):,}  (verbatim)")
    print(f"  test_merged:  {len(df_test):,}  (verbatim)")
    print(f"  total:        {len(merged_train)+len(df_val)+len(df_test):,}")
    print(f"  per-src in train_merged:")
    print(merged_train["_src"].value_counts().to_string())
    print(f"  report -> {out_dir/'merge_report.json'}")


if __name__ == "__main__":
    main()

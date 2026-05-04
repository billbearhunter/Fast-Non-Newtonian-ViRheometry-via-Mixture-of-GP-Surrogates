"""Merge any sub with N < threshold into its nearest neighbor sub by PCA
centroid distance. Operates on an existing partition bank in place.

Workflow:
  For each (gid, bin):
    1. Collect subs in that bin
    2. While any sub has N < threshold:
         - Pick the smallest sub
         - Find nearest other sub by PCA centroid distance
         - Reassign the smallest sub's points to the nearest
         - Recompute centroid of merged sub (mean of new member PCs)
         - Drop the absorbed sub_id
    3. Re-number sub_ids contiguously within the bin

Saves new state.pkl and sub_assignments.csv (overwrites; backup .premerge_bak first).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]


def merge_one_gid(state_dir: Path, threshold: int) -> tuple[int, int]:
    """Returns (n_subs_before, n_subs_after)."""
    with open(state_dir / "state.pkl", "rb") as f:
        st = pickle.load(f)
    df = pd.read_csv(state_dir / "sub_assignments.csv")
    counts = df.groupby("sub_id").size().to_dict()

    n_before = st["n_subs_total"]
    new_bins = []
    sub_remap = {}  # old_sub_id -> new_sub_id
    next_sid = 0

    for b in st["bins"]:
        # Sort subs by current N ascending
        subs = list(b["subs"])
        # Build initial centroids and members
        sub_data = {s["sub_id"]: {
            "centroid": s["centroid_pc"].copy(),
            "shape_id": s["shape_id"],
            "members": (df.sub_id == s["sub_id"]).to_numpy(),
            "N": int(counts.get(s["sub_id"], 0)),
            "mu_z": s["mu_z"], "cov_z": s["cov_z"],
        } for s in subs}

        active = [sid for sid, d in sub_data.items() if d["N"] > 0]
        # iterative merge
        while True:
            if len(active) <= 1:
                break
            tiny = [sid for sid in active if sub_data[sid]["N"] < threshold]
            if not tiny:
                break
            # smallest first
            tiny.sort(key=lambda s: sub_data[s]["N"])
            t = tiny[0]
            t_centroid = sub_data[t]["centroid"]
            # find nearest other in active
            others = [sid for sid in active if sid != t]
            dists = [np.linalg.norm(sub_data[sid]["centroid"] - t_centroid) for sid in others]
            near = others[int(np.argmin(dists))]
            # merge t into near
            t_members = sub_data[t]["members"]
            n_members = sub_data[near]["members"]
            sub_data[near]["members"] = n_members | t_members
            # recompute centroid as weighted average
            t_n, n_n = sub_data[t]["N"], sub_data[near]["N"]
            sub_data[near]["centroid"] = (sub_data[near]["centroid"] * n_n + t_centroid * t_n) / (n_n + t_n)
            # recompute mu_z, cov_z over merged members
            merged_idx = sub_data[near]["members"]
            X = df.loc[merged_idx, ["n", "eta", "sigma_y"]].to_numpy()
            z = np.column_stack([X[:, 0], np.log(np.clip(X[:, 1], 1e-12, None)),
                                  np.log(np.clip(X[:, 2], 1e-12, None))])
            sub_data[near]["mu_z"] = z.mean(axis=0)
            sub_data[near]["cov_z"] = np.cov(z.T) if len(z) > 1 else np.diag([0.04, 0.36, 0.36])
            sub_data[near]["N"] = n_n + t_n
            active.remove(t)

        # build new sub list with re-mapped sub_ids
        new_subs = []
        for sid in active:
            d = sub_data[sid]
            new_sid = next_sid
            sub_remap[sid] = new_sid
            new_subs.append({
                "sub_id": new_sid,
                "bin_id": b["bin_id"],
                "shape_id": d["shape_id"],
                "centroid_pc": d["centroid"].copy(),
                "N": d["N"],
                "mu_z": d["mu_z"],
                "cov_z": d["cov_z"],
                "member_idx": np.where(d["members"])[0],
            })
            next_sid += 1
        new_bins.append({
            "bin_id": b["bin_id"],
            "y8_lo": b["y8_lo"], "y8_hi": b["y8_hi"],
            "n_train": b["n_train"],
            "pca": b["pca"], "scaler": b["scaler"],
            "kmeans_centroids": np.stack([s["centroid_pc"] for s in new_subs]) if new_subs else np.empty((0, 3)),
            "subs": new_subs,
        })

    # Update df.sub_id with the remap
    df["sub_id"] = df.sub_id.map(lambda s: sub_remap.get(int(s), -1))
    new_state = {
        "version": st["version"],
        "gid": st["gid"],
        "length_edges": st["length_edges"],
        "bins": new_bins,
        "n_subs_total": next_sid,
    }
    return new_state, df, n_before, next_sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=PIPE / "Models" / "v10_yshape_v3")
    ap.add_argument("--threshold", type=int, default=100,
                    help="Subs with N < threshold get merged")
    ap.add_argument("--gids", type=str, default="all",
                    help="Comma-separated gids or 'all'")
    a = ap.parse_args()

    if a.gids == "all":
        gid_dirs = sorted(a.bank.glob("state_gid_*"))
    else:
        wanted = {int(g) for g in a.gids.split(",")}
        gid_dirs = [a.bank / f"state_gid_{g}" for g in sorted(wanted)
                     if (a.bank / f"state_gid_{g}").exists()]

    print(f"Merging tiny subs (N<{a.threshold}) in {len(gid_dirs)} gids\n")
    print(f"{'gid':>4s}  {'before':>6s}  {'after':>5s}  {'merged':>6s}")
    print('-' * 30)
    for gd in gid_dirs:
        gid = int(gd.name.split("_")[-1])
        # Backup
        for f in ["state.pkl", "sub_assignments.csv"]:
            src = gd / f
            bak = gd / f"{f}.premerge_bak"
            if src.exists() and not bak.exists():
                shutil.copy(src, bak)
        new_state, new_df, n_before, n_after = merge_one_gid(gd, a.threshold)
        with open(gd / "state.pkl", "wb") as f:
            pickle.dump(new_state, f)
        new_df.to_csv(gd / "sub_assignments.csv", index=False)
        print(f"{gid:>4d}  {n_before:>6d}  {n_after:>5d}  {n_before-n_after:>6d}")


if __name__ == "__main__":
    main()

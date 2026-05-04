"""V10 y-shape partition: Layer 2 + Layer 3 redesign.

Pipeline:
  Layer 1 (gid)         = unchanged geo router on (W, H).
  Layer 2 (length bin)  = K_len quantile bins on y8 per gid (deterministic, no
                          BGM, no junk basins by construction).
  Layer 3 (shape sub)   = K_sh KMeans clusters on PCA(norm_curve) per bin.
                          Each (gid, bin, sub) is one trainable expert.

Output:
  Models/v10_yshape_v1/state_gid_<g>/state.pkl
      {version, gid, length_edges,
       bins: [ {bin_id, y8_lo, y8_hi, n_train, pca, scaler,
                subs: [ {sub_id, shape_id, centroid_pc, N,
                          mu_z, cov_z, X_phys_idx} ] } ] }
  Models/v10_yshape_v1/state_gid_<g>/sub_assignments.csv
      (n, eta, sigma_y, width, height, x_01..x_08, gid, length_bin, sub_id)

Re-uses train+val splits from data/synthetic_splits_clean/. Test set is
NOT touched (used later for evaluation only).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PIPE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import Optimization.libs.engine as V10  # noqa: E402


def norm_curve(Y8: np.ndarray) -> np.ndarray:
    """y(t) shape after normalizing absolute scale.

    Input Y8: (N, 8). Output (N, 7) = (y_2..y_8 - y_1) / y_8.
    Skips frame 1 (contact instant, noise-prone). Divides by y_8 so two
    materials with same shape but different magnitudes map to same point.
    """
    y1 = Y8[:, 0:1]
    y8 = np.clip(Y8[:, 7:8], 1e-6, None)
    return (Y8[:, 1:8] - y1) / y8


def _compute_bin_edges_with_adaptive_merge(
    y8: np.ndarray,
    k_len: int,
    log_y8: bool,
    min_n_per_sub: int,
    merge_threshold: int,
    gid: int,
    verbose: bool = True,
) -> np.ndarray:
    """Plan B: equal-width log-y8 starts with K_len bins; iteratively merge
    sparse adjacent bins until each surviving bin has at least
    `merge_threshold` points (so each bin can support at least 2 KMeans
    clusters of size `min_n_per_sub` each).

    Why merge instead of skip:
      * skipping leaves "holes" in y8 coverage — points falling in skipped
        bin range have no sub to route to at inverse time;
      * merging keeps the log-y8 axis covered end-to-end while avoiding
        starvation of any single bin's GP. The dense main-flow region
        retains finer subdivision; the data-sparse long tail collapses
        into a single bin instead of disappearing.
    """
    if log_y8:
        ly8 = np.log(np.clip(y8, 1e-6, None))
        log_edges = np.linspace(ly8.min(), ly8.max(), k_len + 1)
        edges = np.exp(log_edges)
    else:
        edges = np.quantile(y8, np.linspace(0.0, 1.0, k_len + 1))
    edges[-1] += 1e-9
    edges[0] = max(edges[0] - 1e-9, 0.0)

    # Iterative adaptive merge of sparse adjacent bins
    while True:
        counts = np.array([
            int(((y8 >= edges[b]) & (y8 < edges[b + 1])).sum())
            for b in range(len(edges) - 1)
        ])
        # If all bins meet threshold, or only one bin left, stop
        if len(counts) <= 1 or counts.min() >= merge_threshold:
            break
        # Find smallest bin
        sb = int(np.argmin(counts))
        # Decide which adjacent bin to merge into
        if sb == 0:
            # merge into right neighbor (drop the right edge of bin 0)
            edges = np.delete(edges, 1)
            if verbose:
                print(f"  gid={gid} adaptive-merge: bin 0 ({counts[0]}) -> bin 1")
        elif sb == len(counts) - 1:
            # merge into left neighbor (drop the left edge of last bin)
            edges = np.delete(edges, len(edges) - 2)
            if verbose:
                print(f"  gid={gid} adaptive-merge: bin {sb} ({counts[sb]}) -> bin {sb-1}")
        else:
            # pick smaller adjacent neighbor to merge with
            if counts[sb - 1] <= counts[sb + 1]:
                # merge with left
                edges = np.delete(edges, sb)
                if verbose:
                    print(f"  gid={gid} adaptive-merge: bin {sb} ({counts[sb]}) -> bin {sb-1}")
            else:
                # merge with right
                edges = np.delete(edges, sb + 1)
                if verbose:
                    print(f"  gid={gid} adaptive-merge: bin {sb} ({counts[sb]}) -> bin {sb+1}")
    if verbose:
        final_counts = [int(((y8 >= edges[b]) & (y8 < edges[b + 1])).sum())
                         for b in range(len(edges) - 1)]
        if len(final_counts) < k_len:
            print(f"  gid={gid} bins after merge: {len(final_counts)} (was {k_len}); "
                  f"counts {final_counts}")
    return edges


def partition_one_gid(df_g: pd.DataFrame, gid: int,
                       k_len: int, k_sh_max: int, n_pc: int,
                       min_n_per_sub: int, log_y8: bool = True,
                       merge_threshold: int = -1,
                       rng_seed: int = 7) -> dict:
    """Length log-y8 -> per-bin PCA + KMeans on shape features.

    `log_y8=True` makes the length bins evenly spaced on log(y8) (Plan B
    default). Sparse tail bins are then **adaptively merged** with their
    nearest non-empty neighbour so every surviving bin has at least
    `merge_threshold` points (2 * min_n_per_sub by default), instead of
    being skipped (which used to leave y8 ranges with no sub at all).
    """
    Y = df_g[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
    X_phys = df_g[["n", "eta", "sigma_y", "width", "height"]].to_numpy(dtype=np.float64)
    y8 = Y[:, 7]
    N = len(Y)

    if merge_threshold < 0:
        merge_threshold = 2 * min_n_per_sub  # default: support at least 2 clusters
    edges = _compute_bin_edges_with_adaptive_merge(
        y8, k_len, log_y8, min_n_per_sub, merge_threshold, gid, verbose=True,
    )

    bins_out = []
    sub_assignments = -np.ones(N, dtype=int)
    next_sub_id = 0
    for b in range(len(edges) - 1):
        mask = (y8 >= edges[b]) & (y8 < edges[b + 1])
        idx = np.where(mask)[0]
        if len(idx) < min_n_per_sub:
            # Should rarely happen after adaptive merge — keep as a safety net
            print(f"  gid={gid} bin={b}: only {len(idx)} points (post-merge), skip")
            continue
        nc = norm_curve(Y[idx])  # (N_b, 7)
        scaler = StandardScaler()
        nc_s = scaler.fit_transform(nc)
        n_pc_eff = min(n_pc, nc.shape[0] - 1, nc.shape[1])
        pca = PCA(n_components=n_pc_eff, random_state=rng_seed)
        pcs = pca.fit_transform(nc_s)
        # Adaptive K_sh: at least min_n_per_sub expected per cluster.
        # If too few points, fall back to k=1 (no shape subdivision).
        if len(idx) < 2 * min_n_per_sub:
            k_sh_eff = 1
        else:
            k_sh_eff = max(2, min(k_sh_max, len(idx) // min_n_per_sub))
        if k_sh_eff == 1:
            labels = np.zeros(len(idx), dtype=int)
            centers = pcs.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k_sh_eff, n_init=10, random_state=rng_seed)
            labels = km.fit_predict(pcs)
            centers = km.cluster_centers_.copy()
            # Merge tiny clusters into their nearest neighbor centroid.
            for _ in range(k_sh_eff):
                counts = np.array([(labels == s).sum() for s in range(len(centers))])
                tiny = np.where((counts > 0) & (counts < min_n_per_sub // 2))[0]
                if len(tiny) == 0:
                    break
                t = tiny[0]
                # find nearest non-tiny centroid
                d = np.linalg.norm(centers - centers[t][None, :], axis=1)
                d[t] = np.inf
                d[counts == 0] = np.inf
                d[(counts > 0) & (counts < min_n_per_sub // 2) & (np.arange(len(centers)) != t)] = np.inf
                if not np.isfinite(d).any():
                    break
                near = int(np.argmin(d))
                labels[labels == t] = near
                counts[t] = 0
            # remap labels to be contiguous
            unique = sorted(set(labels.tolist()))
            remap = {old: new for new, old in enumerate(unique)}
            labels = np.array([remap[l] for l in labels])
            centers = np.stack([pcs[labels == s].mean(axis=0)
                                  for s in range(len(unique))])
            k_sh_eff = len(unique)
        subs = []
        for s in range(k_sh_eff):
            members = idx[labels == s]
            if len(members) < min(20, min_n_per_sub // 4):
                continue
            X_local = X_phys[members]
            z = np.column_stack([
                X_local[:, 0],
                np.log(np.clip(X_local[:, 1], 1e-12, None)),
                np.log(np.clip(X_local[:, 2], 1e-12, None)),
            ])
            mu_z = z.mean(axis=0)
            cov_z = np.cov(z.T) if len(z) > 1 else np.diag([0.04, 0.36, 0.36])
            subs.append({
                "sub_id": int(next_sub_id),
                "bin_id": int(b),
                "shape_id": int(s),
                "centroid_pc": centers[s].copy(),
                "N": int(len(members)),
                "mu_z": mu_z,
                "cov_z": cov_z,
                "member_idx": members,
            })
            sub_assignments[members] = next_sub_id
            next_sub_id += 1
        bins_out.append({
            "bin_id": int(b),
            "y8_lo": float(edges[b]),
            "y8_hi": float(edges[b + 1]),
            "n_train": int(len(idx)),
            "pca": {
                "mean": pca.mean_.copy(),
                "components": pca.components_.copy(),
                "n_components": n_pc_eff,
            },
            "scaler": {
                "mean": scaler.mean_.copy(),
                "std": scaler.scale_.copy(),
            },
            "kmeans_centroids": centers.copy(),
            "subs": subs,
        })

    state = {
        "version": "v10_yshape_v1",
        "gid": int(gid),
        "length_edges": edges,
        "bins": bins_out,
        "n_subs_total": int(next_sub_id),
    }
    return state, sub_assignments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path,
                    default=PIPE / "data" / "synthetic_splits_clean" / "train_v10_clean.csv")
    ap.add_argument("--val-csv", type=Path,
                    default=PIPE / "data" / "synthetic_splits_clean" / "val_v10_clean.csv")
    ap.add_argument("--out-bank", type=Path,
                    default=PIPE / "Models" / "v10_yshape_v1")
    ap.add_argument("--k-len", type=int, default=5,
                    help="Number of length quantile bins per gid")
    ap.add_argument("--k-sh-max", type=int, default=5,
                    help="Max KMeans clusters per length bin (adaptive down based on N)")
    ap.add_argument("--n-pc", type=int, default=3,
                    help="Number of PCA components on normalized shape curve")
    ap.add_argument("--min-n-per-sub", type=int, default=80,
                    help="Min points per sub; bins/clusters smaller than this are skipped or merged")
    ap.add_argument("--merge-threshold", type=int, default=-1,
                    help="Plan B adaptive bin merge: each surviving log-y8 bin "
                         "must have at least this many points; sparse adjacent "
                         "bins are merged iteratively. Default -1 = "
                         "2 * min_n_per_sub. Set 0 to disable (legacy "
                         "skip-and-leave behavior).")
    ap.add_argument("--use-val", action="store_true", default=True,
                    help="Include val rows alongside train for partition fitting")
    ap.add_argument("--log-y8", action="store_true", default=True,
                    help="Use log(y8) for length quantile (default; better with heavy tails)")
    ap.add_argument("--linear-y8", dest="log_y8", action="store_false",
                    help="Force linear y8 quantile (legacy)")
    ap.add_argument("--gids", type=str, default="all",
                    help="Comma-separated gid list or 'all' (default)")
    ap.add_argument("--infill-glob", type=str, default=None,
                    help="Optional glob (e.g. 'pipeline_v10/Models/v10_yshape_v1/state_gid_*/"
                         "infill_*.csv') to merge densification CSVs into the "
                         "partition refit. Re-fitting on combined data prevents "
                         "stale centroids when new shape modes appear.")
    args = ap.parse_args()

    geo_router, _xs, _ys = V10.load_runtime()

    df_train = pd.read_csv(args.train_csv)
    if args.use_val and args.val_csv.exists():
        df_val = pd.read_csv(args.val_csv)
        df = pd.concat([df_train, df_val], ignore_index=True)
        print(f"loaded train={len(df_train)}  val={len(df_val)}  total={len(df)}")
    else:
        df = df_train
        print(f"loaded train={len(df)}")

    if args.infill_glob:
        infill_paths = sorted(Path().glob(args.infill_glob))
        if infill_paths:
            keep_cols = [c for c in df.columns]
            extra_dfs = []
            for p in infill_paths:
                d = pd.read_csv(p)
                d = d[[c for c in keep_cols if c in d.columns]]
                extra_dfs.append(d)
                print(f"  + infill {p.name}: {len(d)} rows")
            df = pd.concat([df] + extra_dfs, ignore_index=True)
            print(f"  combined train+val+infill={len(df)}")

    df["gid"] = geo_router.predict(df[["width", "height"]].to_numpy()).astype(int)
    args.out_bank.mkdir(parents=True, exist_ok=True)

    summary = []
    if args.gids != "all":
        wanted_gids = {int(g) for g in args.gids.split(",")}
        df = df[df.gid.isin(wanted_gids)]
        print(f"\nfiltering to gids: {sorted(wanted_gids)}  rows={len(df)}")
    for gid, df_g in df.groupby("gid"):
        df_g = df_g.reset_index(drop=True)
        gd = args.out_bank / f"state_gid_{int(gid)}"
        gd.mkdir(parents=True, exist_ok=True)
        print(f"\n=== gid {gid}: {len(df_g)} points ===")
        t0 = time.time()
        state, assignments = partition_one_gid(
            df_g, int(gid),
            k_len=args.k_len, k_sh_max=args.k_sh_max,
            n_pc=args.n_pc, min_n_per_sub=args.min_n_per_sub,
            log_y8=args.log_y8,
            merge_threshold=args.merge_threshold,
        )
        with open(gd / "state.pkl", "wb") as f:
            pickle.dump(state, f)
        df_g["length_bin"] = -1
        for b in state["bins"]:
            mask = (df_g[[f"x_{i:02d}" for i in range(1, 9)]].iloc[:, 7].to_numpy() >= b["y8_lo"]) \
                   & (df_g[[f"x_{i:02d}" for i in range(1, 9)]].iloc[:, 7].to_numpy() < b["y8_hi"])
            df_g.loc[mask, "length_bin"] = b["bin_id"]
        df_g["sub_id"] = assignments
        df_g.to_csv(gd / "sub_assignments.csv", index=False)

        n_subs = state["n_subs_total"]
        print(f"  -> wrote {n_subs} subs across {len(state['bins'])} bins  ({time.time()-t0:.1f}s)")
        summary.append({
            "gid": int(gid),
            "N": int(len(df_g)),
            "n_bins": int(len(state["bins"])),
            "n_subs": int(n_subs),
            "fit_seconds": float(time.time() - t0),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_bank / "partition_summary.csv", index=False)
    (args.out_bank / "partition_args.json").write_text(json.dumps({
        "k_len": args.k_len, "k_sh_max": args.k_sh_max,
        "n_pc": args.n_pc, "min_n_per_sub": args.min_n_per_sub,
        "use_val": args.use_val,
    }, indent=2))

    print(f"\n{'='*70}")
    print(f"Partition complete. Bank: {args.out_bank}")
    print(f"Total subs across all gids: {summary_df.n_subs.sum()}")
    print(f"Median subs per gid: {summary_df.n_subs.median():.0f}  "
          f"min={summary_df.n_subs.min()}  max={summary_df.n_subs.max()}")


if __name__ == "__main__":
    main()

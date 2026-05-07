"""Master script: re-partition + train + calibrate the 'unused' gids.

The current production gids (0, 10, 24) are y8w=5 / nc_v2 partitioned and
trained.  Other gids fall into two states:

  Missing experts (partition done, GP never trained — n_experts=0):
      3, 4, 6, 7, 8, 9, 23

  Old v1 partition + trained experts (works but pre-y8q-breakthrough):
      1, 2, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22

Per-gid pipeline:
    1. partition_flat_kmeans.py  --y8-weight 5.0 --nc-version v2 --write
    2. train_yshape_subs.py      --force    (~30-60 min each on GPU)
    3. calibrate_subs_holdout.py             (~1-2 min per gid)

Estimated total time: 21 gids × ~50 min = ~17 hours.  Run in background.

Usage:
    # Default — process the 21 unused gids in priority order (missing first)
    python scripts/retrain_unused_gids.py --bank Models/v10_yshape_planB

    # Subset (recommended for testing)
    python scripts/retrain_unused_gids.py --bank Models/v10_yshape_planB --gids 3,4

    # Skip already-done (resume after interruption)
    python scripts/retrain_unused_gids.py --bank Models/v10_yshape_planB --skip-if-current
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Priority order: missing-experts first (most urgent), then v1-partition gids.
DEFAULT_PRIORITY = [
    # No experts at all
    3, 4, 6, 7, 8, 9, 23,
    # v1 partition (functional but pre-y8q)
    1, 2, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
]


def is_current(gid_dir: Path) -> bool:
    """Returns True if gid is already y8w=5 / nc_v2 with experts."""
    sd = gid_dir / "state.pkl"
    if not sd.exists():
        return False
    try:
        with open(sd, "rb") as f:
            d = pickle.load(f)
        if d.get("nc_version") != "v2":
            return False
        if abs(float(d.get("y8_weight", 0)) - 5.0) > 1e-6:
            return False
        exp_dir = gid_dir / "experts"
        if not exp_dir.exists() or len(list(exp_dir.glob("sub_*.pt"))) == 0:
            return False
        return True
    except Exception:
        return False


def run_step(label: str, cmd: list[str], log_path: Path, env=None) -> tuple[bool, float]:
    print(f"\n[{time.strftime('%H:%M:%S')}] [{label}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== {label}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(REPO), env=env)
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    print(f"[{time.strftime('%H:%M:%S')}] [{label}] {'OK' if ok else 'FAIL'} (rc={proc.returncode}) in {elapsed:.1f}s", flush=True)
    return ok, elapsed


def process_gid(bank: Path, gid: int, log_path: Path, env) -> dict:
    print(f"\n{'='*72}")
    print(f"  GID {gid}  start  {time.strftime('%H:%M:%S')}")
    print(f"{'='*72}", flush=True)
    out = {"gid": gid, "partition_ok": False, "train_ok": False, "calibrate_ok": False,
           "partition_s": 0.0, "train_s": 0.0, "calibrate_s": 0.0}

    # Step 1: partition with v2 + y8w=5
    out["partition_ok"], out["partition_s"] = run_step(
        f"gid={gid} partition",
        [sys.executable, "scripts/partition_flat_kmeans.py",
         "--bank", str(bank), "--gid", str(gid),
         "--y8-weight", "5.0", "--nc-version", "v2", "--write"],
        log_path, env=env,
    )
    if not out["partition_ok"]:
        return out

    # Step 2: train experts (--force overrides any stale .pt)
    out["train_ok"], out["train_s"] = run_step(
        f"gid={gid} train",
        [sys.executable, "-m", "surrogate.train_yshape_subs",
         "--bank", str(bank), "--gids", str(gid), "--force"],
        log_path, env=env,
    )
    if not out["train_ok"]:
        return out

    # Step 3: calibrate
    out["calibrate_ok"], out["calibrate_s"] = run_step(
        f"gid={gid} calibrate",
        [sys.executable, "scripts/calibrate_subs_holdout.py",
         "--bank", str(bank), "--gids", str(gid)],
        log_path, env=env,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, default=REPO / "Models" / "v10_yshape_planB")
    ap.add_argument("--gids", type=str, default=None,
                    help="Comma-separated gids; default = priority list")
    ap.add_argument("--skip-if-current", action="store_true",
                    help="Skip gids that already pass is_current() check")
    ap.add_argument("--log-dir", type=Path,
                    default=REPO / "Models" / "v10_yshape_planB" / "_retrain_logs")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"retrain_unused_gids_{time.strftime('%Y%m%d_%H%M%S')}.log"

    if args.gids:
        gids = [int(g) for g in args.gids.split(",") if g.strip()]
    else:
        gids = DEFAULT_PRIORITY

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"=== retrain_unused_gids ===  bank={args.bank.name}  log={log_path}")
    print(f"  Queue: {gids}")
    if args.skip_if_current:
        before = list(gids)
        gids = [g for g in gids if not is_current(args.bank / f"state_gid_{g}")]
        print(f"  After skip-if-current: {gids}  (skipped: {sorted(set(before)-set(gids))})")

    summary = []
    t_global = time.time()
    for gid in gids:
        try:
            res = process_gid(args.bank, gid, log_path, env)
        except Exception as e:
            res = {"gid": gid, "error": str(e)}
            print(f"[gid={gid}] EXCEPTION: {e}", flush=True)
        summary.append(res)
        # Print running totals
        total_min = (time.time() - t_global) / 60
        done_ok = sum(1 for r in summary if r.get("calibrate_ok"))
        print(f"\n  Running progress: {done_ok}/{len(summary)} gids done OK; "
              f"total elapsed {total_min:.1f} min", flush=True)

    # Final summary
    print("\n" + "=" * 72)
    print("=== FINAL SUMMARY ===")
    print(f"{'gid':>4}  {'part':>4}  {'train':>5}  {'calib':>5}  {'partS':>7}  {'trainS':>8}  {'calibS':>7}")
    for r in summary:
        gid = r.get("gid")
        if "error" in r:
            print(f"{gid:>4}  EXCEPTION: {r['error']}")
            continue
        print(f"{gid:>4}  {'✓' if r['partition_ok'] else '✗':>4}  "
              f"{'✓' if r['train_ok'] else '✗':>5}  "
              f"{'✓' if r['calibrate_ok'] else '✗':>5}  "
              f"{r['partition_s']:>7.1f}  {r['train_s']:>8.1f}  {r['calibrate_s']:>7.1f}")
    print(f"\nTotal time: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()

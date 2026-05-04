"""Clean image-diff between two simulation runs (same renderer / particle
radius / camera), plus silhouette-IoU metric.

Use-case: the ref images shipped in data/ref_<...>/config_XX.png were
rendered with a *different* particle radius than the current Simulation
code. Diffing a fresh sim against those refs picks up a constant rim
offset that is **not** a physics-fit error. The clean protocol is:

  1. Re-render the ground-truth (n, eta, sigma_y) with the current code
     on each ref's (W, H, camera). Call this GT_run.
  2. Re-render the recovered θ likewise. Call this REC_run.
  3. Diff REC_run vs GT_run (same radius) — this ISOLATES the physics
     error.
  4. Also compute the silhouette-IoU (binary mask overlap) as a scalar.

Usage:
    python -m Simulation.clean_diff \\
        --gt-dir  results/gt_rerender_20260421/gt_30x25 \\
        --rec-dir results/rbcm_2setup_20260421/v2_stage2_3.0_2.5 \\
        --out-dir results/rbcm_2setup_20260421/v2_stage2_3.0_2.5_cleandiff \\
        --amplify 5.0
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


_CFG_RE = re.compile(r"^config_(\d+)\.png$", re.IGNORECASE)


def _find_configs(d: Path) -> List[Tuple[str, Path]]:
    """Walk dir, return [(idx, path), ...] for every config_XX.png found."""
    out = []
    for p in d.rglob("config_*.png"):
        m = _CFG_RE.match(p.name)
        if m:
            out.append((m.group(1), p))
    out.sort()
    return out


def _binarize(img: Image.Image, bg_thresh: int = 10) -> np.ndarray:
    """Foreground mask: pixel is FG if max(R,G,B) > bg_thresh. The ref
    images have a near-black background, so this is robust.
    """
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    return (arr.max(axis=2) > bg_thresh)


def compute_diff_and_iou(gt_path: Path, rec_path: Path, out_path: Path,
                        amplify: float = 5.0,
                        bg_thresh: int = 10) -> dict:
    gt = Image.open(gt_path).convert("RGB")
    rec = Image.open(rec_path).convert("RGB")
    if gt.size != rec.size:
        rec = rec.resize(gt.size, Image.LANCZOS)

    gt_arr  = np.array(gt,  dtype=np.int16)
    rec_arr = np.array(rec, dtype=np.int16)

    # Amplified pixel-diff image
    diff = np.abs(rec_arr - gt_arr).astype(np.float32)
    diff_img = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    Image.fromarray(diff_img, mode="RGB").save(out_path)

    # Silhouette IoU (binary-mask overlap)
    m_gt  = _binarize(gt,  bg_thresh)
    m_rec = _binarize(rec, bg_thresh)
    inter = np.logical_and(m_gt, m_rec).sum()
    union = np.logical_or (m_gt, m_rec).sum()
    iou = float(inter) / float(union) if union > 0 else float("nan")

    # Per-pixel RMS (raw, no amplify)
    rms = float(np.sqrt(np.mean(diff ** 2)))
    # Fraction of pixels with any visible disagreement (>5/255 on any channel)
    frac_diff = float(((rec_arr - gt_arr).max(axis=2) > 5).mean())

    return {
        "gt_path":  str(gt_path),
        "rec_path": str(rec_path),
        "iou":      iou,
        "rms":      rms,
        "frac_diff_5": frac_diff,
        "mask_gt_px":  int(m_gt.sum()),
        "mask_rec_px": int(m_rec.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir",  required=True,
                    help="GT re-render directory (same particle radius as "
                         "--rec-dir). Contains config_XX.png.")
    ap.add_argument("--rec-dir", required=True,
                    help="Recovered-theta sim directory.")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for clean_snapdiff_XX.png + CSV.")
    ap.add_argument("--amplify", type=float, default=5.0)
    ap.add_argument("--bg-thresh", type=int, default=10,
                    help="Background pixel threshold for silhouette mask.")
    args = ap.parse_args()

    gt_dir  = Path(args.gt_dir)
    rec_dir = Path(args.rec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_cfgs  = {idx: p for idx, p in _find_configs(gt_dir)}
    rec_cfgs = {idx: p for idx, p in _find_configs(rec_dir)}
    shared = sorted(set(gt_cfgs.keys()) & set(rec_cfgs.keys()))
    if not shared:
        print(f"[clean_diff] ERROR: no common config indices between "
              f"{gt_dir} and {rec_dir}")
        return

    rows = []
    for idx in shared:
        out_path = out_dir / f"clean_snapdiff_{idx}.png"
        r = compute_diff_and_iou(
            gt_cfgs[idx], rec_cfgs[idx], out_path,
            amplify=args.amplify, bg_thresh=args.bg_thresh,
        )
        r["frame"] = idx
        rows.append(r)
        print(f"  frame {idx}: IoU={r['iou']:.4f}  RMS={r['rms']:.2f}  "
              f"frac_diff>5={r['frac_diff_5']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "clean_diff_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  mean IoU = {df['iou'].mean():.4f}  "
          f"(min={df['iou'].min():.4f}, max={df['iou'].max():.4f})")
    print(f"  mean RMS = {df['rms'].mean():.2f}")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Recalibrate camera using an existing (possibly PS-cleaned) config_00.png.

Skips video extraction / binarization / ROI picking — just:
  1. ChArUco on bg image → K, R_init, t_init (prior)
  2. refine_extrinsic_edge on config_00 → refined R, t
  3. Save camera_params.xml + Background_mask.png + diff_*.png

Two-step workflow:
  Step A (once):  python Calibration/prepare_configs.py --data_dir <DIR>
  Step B (repeat as you clean config_00 in PS):
                  python Calibration/recalibrate_only.py --data_dir <DIR>

Auto-detection (from --data_dir):
  - bg_img      : first *.JPG / *.jpg in data_dir
  - configs_dir : <data_dir>/output  (override with --configs_dir)
  - fluid_w / h : parsed from settings.xml (override with --fluid_w/h)

Example (with left+top+bottom corner-emphasis):
    python Calibration/recalibrate_only.py --data_dir data/ref_Sweet_4.5_4.5_1 \\
        --left_weight 20 --top_weight 20 --bottom_weight 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from pipeline import (
    calibrate, save_xml, refine_extrinsic_edge,
    render_mask_KRt, render_background_KRt, diff_visual, diff_binary,
    _theta_to_K, _camera_axes, _parse_settings_xml,
)


def main():
    p = argparse.ArgumentParser(
        description="Recalibrate camera from an existing config_00.png "
                    "(Step B of the two-step workflow — see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", required=True,
                   help="Experiment dir (must contain IMG_*.JPG + settings.xml)")
    p.add_argument("--configs_dir", default=None,
                   help="Dir containing config_00.png (target). "
                        "Default: <data_dir>/output")
    p.add_argument("--bg_img", default=None,
                   help="Background image path (default: auto-detect *.JPG in data_dir)")
    p.add_argument("--fluid_w", type=float, default=None,
                   help="Container width in cm. Default: read from settings.xml <setup W=...>")
    p.add_argument("--fluid_h", type=float, default=None,
                   help="Container height in cm. Default: read from settings.xml <setup H=...>")
    p.add_argument("--safety_rot_deg", type=float, default=5.0,
                   help="Rotation drift limit (default: 5.0 — relaxed from 3.0)")
    p.add_argument("--safety_t_cm", type=float, default=5.0,
                   help="Translation drift limit (default: 5.0)")
    p.add_argument("--left_weight", type=float, default=1.0,
                   help="Multiplier for left-side pixels in chamfer+IoU "
                        "(default: 1.0 = uniform; try 5-20 when right side "
                        "has fluid-tongue contamination)")
    p.add_argument("--left_frac", type=float, default=0.5,
                   help="Fraction of target bbox width that counts as "
                        "'left' (default: 0.5)")
    p.add_argument("--top_weight", type=float, default=1.0,
                   help="Multiplier for top-side pixels in chamfer+IoU "
                        "(default: 1.0 = uniform). Combines multiplicatively "
                        "with left_weight, so top-left corner gets "
                        "left_weight*top_weight boost.")
    p.add_argument("--top_frac", type=float, default=0.5,
                   help="Fraction of target bbox height that counts as "
                        "'top' (default: 0.5)")
    p.add_argument("--bottom_weight", type=float, default=1.0,
                   help="Multiplier for bottom-side pixels in chamfer+IoU "
                        "(default: 1.0 = uniform). Combines multiplicatively "
                        "with left_weight, so bottom-left corner gets "
                        "left_weight*bottom_weight boost.")
    p.add_argument("--bottom_frac", type=float, default=0.5,
                   help="Fraction of target bbox height (measured from the "
                        "bottom) that counts as 'bottom' (default: 0.5)")
    p.add_argument("--force-refined", action="store_true",
                   help="Accept the refined extrinsic even when the global "
                        "(unweighted) IoU drops below ChArUco. Use when "
                        "ChArUco itself is tilted and you trust the weighted "
                        "edge alignment. Safety clamp still applies.")
    p.add_argument("--start-from-xml", action="store_true",
                   help="Use the existing camera_params.xml in <configs_dir> "
                        "as the prior (skip ChArUco re-fit). Use after manual "
                        "tweaks: optimizer continues from the hand-aligned "
                        "state and refines further.")
    p.add_argument("--search-rot-deg", type=float, default=2.5,
                   help="Phase-1/2 optimizer search range in rotation (deg). "
                        "Default 2.5. Lower (e.g. 0.5) keeps the refined R "
                        "close to the prior — useful when the optimizer's "
                        "global minimum is in a visually wrong basin.")
    p.add_argument("--search-t-cm", type=float, default=2.0,
                   help="Phase-1/2 optimizer search range in translation (cm). "
                        "Default 2.0.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[ERROR] --data_dir {data_dir} not found")

    # Auto-resolve configs_dir
    cfg_dir = Path(args.configs_dir) if args.configs_dir else data_dir / "output"
    if not cfg_dir.is_dir():
        sys.exit(f"[ERROR] configs_dir {cfg_dir} not found "
                 f"(run prepare_configs.py first)")
    print(f"[configs]  {cfg_dir}")

    # Auto-detect bg image
    bg_img_path = args.bg_img
    if bg_img_path is None:
        jpgs = sorted(data_dir.glob("*.JPG")) + sorted(data_dir.glob("*.jpg"))
        if not jpgs:
            sys.exit(f"[ERROR] No *.JPG in {data_dir} — use --bg_img")
        bg_img_path = str(jpgs[0])
    print(f"[bg_img]   {bg_img_path}")

    # Auto-detect fluid_w / fluid_h from settings.xml
    if args.fluid_w is None or args.fluid_h is None:
        sx = data_dir / "settings.xml"
        if not sx.is_file():
            sx = cfg_dir / "settings.xml"
        parsed = _parse_settings_xml(str(sx)) if sx.is_file() else {}
        if args.fluid_w is None:
            args.fluid_w = parsed.get("W")
        if args.fluid_h is None:
            args.fluid_h = parsed.get("H")
        if args.fluid_w is None or args.fluid_h is None:
            sys.exit(f"[ERROR] Cannot read W/H from {sx}. "
                     f"Pass --fluid_w / --fluid_h explicitly.")
        print(f"[fluid]    W={args.fluid_w}cm  H={args.fluid_h}cm  (from {sx.name})")

    cfg00 = cfg_dir / "config_00.png"
    if not cfg00.is_file():
        sys.exit(f"[ERROR] {cfg00} not found")
    print(f"[target]   {cfg00}  (PS-cleaned)")

    target = cv2.imread(str(cfg00), cv2.IMREAD_GRAYSCALE)
    tgt_H, tgt_W = target.shape
    print(f"[target]   {tgt_W}x{tgt_H}, black pixels = "
          f"{int((target < 128).sum())} ({(target < 128).mean()*100:.2f}%)")

    fw_m = args.fluid_w / 100.0
    fh_m = args.fluid_h / 100.0

    # ── Step 4a: prior — ChArUco re-fit OR existing camera_params.xml ──
    if args.start_from_xml:
        from scipy.spatial.transform import Rotation as _R
        import xml.etree.ElementTree as _ET
        xml_path = cfg_dir / "camera_params.xml"
        if not xml_path.is_file():
            sys.exit(f"[ERROR] {xml_path} not found — "
                     f"--start-from-xml needs an existing camera_params.xml")
        cam = _ET.parse(xml_path).getroot().find("camera")
        eye_cm = np.array([float(v) for v in cam.attrib["eyepos"].split()])
        qw, qx, qy, qz = [float(v) for v in cam.attrib["quat"].split()]
        fov_deg = float(cam.attrib["fov"])
        # Inverse of save_xml: q -> rot_our, then R_mat = [C_x; -C_y; -C_z]
        rot_our = _R.from_quat([qx, qy, qz, qw]).as_matrix()
        R_init = np.vstack([rot_our[:, 0], -rot_our[:, 1], -rot_our[:, 2]])
        eye = eye_cm / 100.0
        t_init = -R_init @ eye
        # Image size from target (config_00); focal length from fov
        img_W, img_H = tgt_W, tgt_H
        f = img_H / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        K_init = np.array([[f, 0, img_W/2.0],
                           [0, f, img_H/2.0],
                           [0, 0, 1.0]])
        print(f"\n[calib] Step 4a: prior from existing camera_params.xml "
              f"(skipping ChArUco)")
        print(f"[calib] eye={eye_cm} cm, fov={fov_deg:.3f}°")
    else:
        print("\n[calib] Step 4a: ChArUco on background image")
        theta0, img_W, img_H = calibrate(bg_img_path, fw_m, fh_m)
        K_init = _theta_to_K(theta0, img_W, img_H)
        eye, C_x, C_y, C_z, _s = _camera_axes(theta0)
        R_init = np.vstack([C_x, -C_y, -C_z])
        t_init = -R_init @ eye
        print(f"[calib] ChArUco eye={eye}, |t|={np.linalg.norm(t_init):.3f}")

    # ── Step 4b: edge refinement ────────────────────────────────────
    print(f"\n[calib] Step 4b: edge refinement "
          f"(safety limits: {args.safety_rot_deg}°/{args.safety_t_cm}cm)")
    result = refine_extrinsic_edge(
        target, K_init, R_init, t_init,
        fw_m, fh_m, tgt_W, tgt_H,
        safety_rot_deg=args.safety_rot_deg,
        safety_t_cm=args.safety_t_cm,
        left_weight=args.left_weight,
        left_frac=args.left_frac,
        top_weight=args.top_weight,
        top_frac=args.top_frac,
        bottom_weight=args.bottom_weight,
        bottom_frac=args.bottom_frac,
        search_rot_deg=args.search_rot_deg,
        search_t_cm=args.search_t_cm,
    )
    if result is None:
        sys.exit("[ERROR] refine_extrinsic_edge failed")

    c0 = result["cost_before"]
    c1 = result["cost_after"]
    R_ref, t_ref = result["R"], result["t"]

    # Did pipeline.py's internal safety clamp fire?  When it does, it returns
    # R_init / t_init verbatim (so R_ref is literally R_init).
    safety_clamped = (np.allclose(R_ref, R_init) and
                      np.allclose(np.asarray(t_ref).flatten(),
                                  t_init.flatten()))

    # Compare the real final metric (unweighted IoU) between ChArUco and
    # refined — chamfer is misleading when Phase 2 optimises WEIGHTED IoU,
    # because the weighted objective routinely raises chamfer while
    # boosting the left/top-corner alignment the user cares about.
    def _iou_unweighted(K_, R_, t_):
        m = render_mask_KRt(K_, R_, t_, fw_m, fh_m, tgt_W, tgt_H)
        fg_m = m < 128
        fg_t = target < 128
        inter = np.logical_and(fg_m, fg_t).sum()
        union = np.logical_or(fg_m, fg_t).sum()
        return inter / union if union > 0 else 0.0

    iou_charuco = _iou_unweighted(K_init, R_init, t_init)
    iou_refined = (iou_charuco if safety_clamped
                   else _iou_unweighted(K_init, R_ref, t_ref))

    if safety_clamped:
        print("[calib] pipeline safety clamp fired — keeping ChArUco")
        K, R_mat, t = K_init, R_init, t_init
    elif args.force_refined or iou_refined >= iou_charuco:
        K, R_mat, t = K_init, R_ref, t_ref
        # Compute drift info
        eye_bg  = -R_init.T @ t_init
        eye_new = -R_mat.T @ t
        delta = (eye_new - eye_bg) * 100.0
        dist  = np.linalg.norm(delta)
        R_delta = R_mat @ R_init.T
        angle_deg = np.degrees(np.arccos(np.clip(
            (np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"[calib] ACCEPTED refinement:")
        print(f"         IoU     {iou_charuco:.4f} → {iou_refined:.4f}")
        print(f"         chamfer {c0:.3f} → {c1:.3f} px "
              f"(chamfer may rise under weighted IoU — OK)")
        print(f"         Δpos = ({delta[0]:+.3f}, {delta[1]:+.3f}, "
              f"{delta[2]:+.3f}) cm |Δ|={dist:.3f}")
        print(f"         Δrot = {angle_deg:.3f}°")
    else:
        print(f"[calib] refinement IoU ({iou_refined:.4f}) < ChArUco "
              f"({iou_charuco:.4f}) — keeping ChArUco")
        K, R_mat, t = K_init, R_init, t_init

    # ── Step 4c: save ───────────────────────────────────────────────
    out_xml = cfg_dir / "camera_params.xml"
    save_xml(K, R_mat, t, img_W, img_H, str(out_xml))

    mask = render_mask_KRt(K, R_mat, t, fw_m, fh_m, tgt_W, tgt_H)
    cv2.imwrite(str(cfg_dir / "Background_mask.png"), mask)
    cv2.imwrite(str(cfg_dir / "diff_combined.png"), diff_visual(mask, target))
    cv2.imwrite(str(cfg_dir / "diff_binary.png"),   diff_binary(mask, target))

    bg_img = cv2.imread(bg_img_path)
    if bg_img is not None:
        bg_out = render_background_KRt(bg_img, K, R_mat, t, fw_m, fh_m,
                                       out_w=tgt_W, out_h=tgt_H)
        cv2.imwrite(str(cfg_dir / "Background.png"), bg_out)

    fg_mask = (mask < 128).astype(np.uint8)
    fg_tgt  = (target < 128).astype(np.uint8)
    inter = np.logical_and(fg_mask, fg_tgt).sum()
    union = np.logical_or(fg_mask, fg_tgt).sum()
    iou = inter / union if union > 0 else 0.0
    print(f"\n[calib] ✓ Done. IoU(rendered vs config_00) = {iou:.4f}")
    print(f"         camera_params.xml → {out_xml}")


if __name__ == "__main__":
    main()

"""Render MPM-direct .obj files to silhouette masks per-frame and compute
SIGGRAPH-style snapdiff vs the actual binarized config_NN.png.

For each frame N (1..8):
  - Load <holdout_dir>/holdout_mpm/<params>/config_NN.obj
  - Project mesh vertices to image coords using <holdout_dir>/output/camera_params.xml
  - Rasterize all triangles → predicted silhouette
  - Cube container wireframe is also drawn (since the rigid container masks the
    OBJ in real video too; pixels inside cube + outside fluid are still mask=0)
  - Diff against <holdout_dir>/config_NN.png:
      RED   = predicted only (sim says fluid, actual says not)
      BLUE  = actual only    (actual says fluid, sim says not)
      BLACK = overlap (correct)
      GRAY  = neither

Usage:
    python scripts/snapdiff_mpm.py --holdout-dir freshness/data/ref_Cream_3.2_6.1_3
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Calibration"))

from pipeline import (
    _parse_settings_xml,
    _cube_verts,
    _CUBE_FACES,
    _FACE_NORMALS,
    _project_KRt,
)


def load_obj_vertices_faces(path: Path):
    """Minimal OBJ parser: returns (V, F)."""
    V = []
    F = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    # Handle 'a/b/c' or 'a'
                    idx.append(int(tok.split("/")[0]) - 1)  # OBJ 1-indexed
                # Triangulate fan (most OBJ are triangles already)
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def quat_to_R_mat(qw, qx, qy, qz):
    rot_our = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.vstack([rot_our[:, 0], -rot_our[:, 1], -rot_our[:, 2]])


def rasterize_mesh(V_world, F, K, R_mat, t, img_w, img_h):
    """Project vertices, draw filled triangles → binary silhouette.

    Handles vertices behind the camera (negative depth) and NaN-projections
    by clipping/dropping affected triangles.
    """
    # Compute camera-space depth to filter behind-camera vertices.
    V_cam = (R_mat @ V_world.T).T + t.reshape(1, 3)  # (N, 3)
    z_cam = V_cam[:, 2]
    pv = _project_KRt(V_world, K, R_mat, t)  # (N, 2)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    INT_MAX_CLIP = 100000
    for tri in F:
        # Skip if any vertex behind camera (negative z in OpenCV convention) or NaN.
        if (z_cam[tri] <= 0).any():
            continue
        pts = pv[tri]
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            continue
        # Bounding box reject (outside image far)
        if (pts[:, 0].min() > img_w + 200 or pts[:, 0].max() < -200 or
            pts[:, 1].min() > img_h + 200 or pts[:, 1].max() < -200):
            continue
        # Clip extreme values before cast to avoid overflow on int32 conversion.
        pts_clip = np.clip(pts, -INT_MAX_CLIP, INT_MAX_CLIP).astype(np.int32)
        cv2.fillConvexPoly(mask, pts_clip, 255)
    return mask  # 255 = predicted fluid


def render_cube_mask(K, R_mat, t, fw_m, fh_m, eye_m, img_w, img_h):
    """Render filled silhouette of the container cube — used as a 'hard
    container' constraint (sim can't show fluid outside the cube even if mesh
    pokes through). Returns 255 where cube is."""
    verts = _cube_verts(fw_m, fh_m)
    pv = _project_KRt(verts, K, R_mat, t).astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for fi, face in enumerate(_CUBE_FACES):
        if _FACE_NORMALS[fi] @ (eye_m - verts[face].mean(axis=0)) > 0:
            cv2.fillConvexPoly(mask, pv[face], 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dir", type=Path, required=True)
    ap.add_argument("--mpm-subdir", type=str, default=None,
                    help="Specific MPM run subdir name (default: first found)")
    args = ap.parse_args()

    data_dir = args.holdout_dir
    cfg_dir = data_dir / "output"

    # Find MPM run dir
    mpm_root = data_dir / "holdout_mpm"
    if not mpm_root.is_dir():
        sys.exit(f"[error] {mpm_root} not found — run Simulation/main.py first")
    if args.mpm_subdir:
        run_dir = mpm_root / args.mpm_subdir
    else:
        subs = [p for p in mpm_root.iterdir() if p.is_dir()]
        if not subs:
            sys.exit(f"[error] No subdirs in {mpm_root}")
        run_dir = subs[0]
    print(f"[load] MPM run: {run_dir}")

    # Camera
    cam = ET.parse(cfg_dir / "camera_params.xml").getroot().find("camera")
    eye_cm = np.array([float(v) for v in cam.attrib["eyepos"].split()])
    qw, qx, qy, qz = [float(v) for v in cam.attrib["quat"].split()]
    fov_deg = float(cam.attrib["fov"])
    R_mat = quat_to_R_mat(qw, qx, qy, qz)
    eye_m = eye_cm / 100.0
    t_m = -R_mat @ eye_m

    # Geometry
    settings = _parse_settings_xml(str(data_dir / "settings.xml"))
    fw_m = settings["W"] / 100.0
    fh_m = settings["H"] / 100.0

    # Intrinsics
    cfg0 = cv2.imread(str(cfg_dir / "config_00.png"), cv2.IMREAD_GRAYSCALE)
    H_img, W_img = cfg0.shape
    f = H_img / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0, W_img/2.0], [0, f, H_img/2.0], [0, 0, 1.0]])

    # Cube mask (used as confinement reference, not subtracted from prediction)
    cube_mask = render_cube_mask(K, R_mat, t_m, fw_m, fh_m, eye_m, W_img, H_img)
    cv2.imwrite(str(cfg_dir / "snapdiff_cube_mask.png"), cube_mask)

    # Per-frame snapdiff
    summary = []
    for N in range(0, 9):
        obj_path = run_dir / f"config_{N:02d}.obj"
        cfg_path = cfg_dir / f"config_{N:02d}.png"
        if not obj_path.is_file() or not cfg_path.is_file():
            continue
        V, F = load_obj_vertices_faces(obj_path)
        if len(F) == 0:
            print(f"[warn] frame {N}: OBJ has no faces, skipping")
            continue
        # OBJ vertices are in cm; R/t expect meters.
        V_m = V / 100.0
        pred_mask = rasterize_mesh(V_m, F, K, R_mat, t_m, W_img, H_img)
        # Predicted: fluid where mesh projects to.
        actual_gray = cv2.imread(str(cfg_path), cv2.IMREAD_GRAYSCALE)
        actual_mask = (actual_gray < 128).astype(np.uint8) * 255

        # IoU + signed diff
        pred_b = pred_mask > 128
        act_b  = actual_mask > 128
        inter = int(np.logical_and(pred_b, act_b).sum())
        union = int(np.logical_or(pred_b, act_b).sum())
        iou = inter / union if union > 0 else 0.0

        # Snapdiff RGB
        diff = np.full((H_img, W_img, 3), 200, dtype=np.uint8)  # gray base
        only_pred = pred_b & ~act_b
        only_act  = act_b  & ~pred_b
        overlap   = pred_b & act_b
        diff[only_pred] = [0, 0, 255]   # red = sim only
        diff[only_act]  = [255, 0, 0]   # blue = actual only
        diff[overlap]   = [0, 0, 0]     # black = correct
        # Label
        cv2.putText(diff, f"frame {N}  IoU={iou:.3f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(diff, "RED=sim only  BLUE=actual only  BLACK=overlap",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = cfg_dir / f"snapdiff_{N:02d}.png"
        cv2.imwrite(str(out), diff)
        # Also save the raw predicted silhouette
        cv2.imwrite(str(cfg_dir / f"snapdiff_pred_{N:02d}.png"), pred_mask)
        summary.append((N, iou, inter, union))
        print(f"  frame {N:02d}  IoU={iou:.4f}  pred={int(pred_b.sum())}  actual={int(act_b.sum())}")

    if summary:
        ious = [s[1] for s in summary]
        print(f"\n[summary] frames={len(ious)}  IoU mean={np.mean(ious):.4f}  median={np.median(ious):.4f}  min={min(ious):.4f}  max={max(ious):.4f}")


if __name__ == "__main__":
    main()

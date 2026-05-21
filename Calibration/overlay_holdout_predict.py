"""Visualise hold-out joint-θ̂ prediction vs actual mask, frame by frame.

Reads holdout_forward.json (predicted y_pred per frame, plus the actual y_obs)
and the existing config_NN.png + camera_params.xml in <holdout_dir>/output/.

For each frame N=1..8 draws on top of config_NN:
  - GREEN cube wireframe (where container should be)
  - RED   actual mask edge (binarized config — what camera saw)
  - YELLOW vertical line at x = W + y_obs[N-1]  (actual front, world units → projected)
  - CYAN   vertical line at x = W + y_pred[N-1] (joint θ̂ prediction → projected)

Saves <holdout_dir>/output/holdout_overlay_NN.png.

Usage:
    python Calibration/overlay_holdout_predict.py --holdout-dir <ref_dir>
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from pipeline import (
    _parse_settings_xml,
    _cube_verts,
    _CUBE_FACES,
    _FACE_NORMALS,
    _project_KRt,
)


def quat_to_R_mat(qw, qx, qy, qz):
    rot_our = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.vstack([rot_our[:, 0], -rot_our[:, 1], -rot_our[:, 2]])


def project_front_line(K, R_mat, t_m, x_world_m, fw_m, fh_m, depth_m=0.04):
    """Project a vertical line at x=x_world_m (world units, metres)
    spanning the fluid extent (z=0..depth, y=0..fh) into image coords.
    Returns the two endpoint pixel coords (top, bottom)."""
    # The fluid front line — at given x_world, y from 0 to fh, z from 0 to depth.
    # We want the visible silhouette: top-back edge and bottom-front edge.
    # For dam-break left wall convention: x grows rightward. The front is the
    # vertical edge at the front (z=0) of the visible face.
    # Simplification: just draw at z=0 (front), y=0..fh.
    pts3d = np.array([
        [x_world_m, 0.0,     0.0],          # bottom front
        [x_world_m, fh_m,    0.0],          # top front
        [x_world_m, fh_m,    depth_m],      # top back
        [x_world_m, 0.0,     depth_m],      # bottom back
    ])
    pv = _project_KRt(pts3d, K, R_mat, t_m)
    return pv  # (4, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dir", type=Path, required=True)
    ap.add_argument("--line-width", type=int, default=2)
    args = ap.parse_args()

    data_dir = args.holdout_dir
    cfg_dir = data_dir / "output"

    # Load forward predict
    fpath = data_dir / "holdout_forward.json"
    if not fpath.is_file():
        sys.exit(f"[error] {fpath} not found — run scripts/holdout_forward.py first")
    fwd = json.loads(fpath.read_text())
    y_obs = np.asarray(fwd["y_obs"], dtype=np.float64)   # 8 frames
    y_pred = np.asarray(fwd["y_pred"], dtype=np.float64)
    print(f"[load] holdout_forward.json: 8 frames y_obs/y_pred")

    # Load camera
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
    depth_m = (settings.get("depth") or 4.0) / 100.0

    # Intrinsics (from first config)
    cfg0 = cv2.imread(str(cfg_dir / "config_00.png"), cv2.IMREAD_GRAYSCALE)
    H_img, W_img = cfg0.shape
    f = H_img / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0, W_img/2.0], [0, f, H_img/2.0], [0, 0, 1.0]])

    # Cube wireframe (once)
    verts_3d = _cube_verts(fw_m, fh_m)
    pv_cube = _project_KRt(verts_3d, K, R_mat, t_m)
    visible_face_idx = []
    for fi, face in enumerate(_CUBE_FACES):
        center = verts_3d[face].mean(axis=0)
        if _FACE_NORMALS[fi] @ (eye_m - center) > 0:
            visible_face_idx.append(fi)

    def draw_cube(canvas, color):
        for fi in visible_face_idx:
            face = _CUBE_FACES[fi]
            corners = pv_cube[face].astype(int)
            for j in range(len(corners)):
                p0 = tuple(corners[j])
                p1 = tuple(corners[(j + 1) % len(corners)])
                cv2.line(canvas, p0, p1, color, args.line_width, cv2.LINE_AA)

    # For each frame N (1..8), draw both front lines on config_N
    n_written = 0
    for N in range(1, 9):
        cfg_path = cfg_dir / f"config_{N:02d}.png"
        if not cfg_path.is_file():
            continue
        gray = cv2.imread(str(cfg_path), cv2.IMREAD_GRAYSCALE)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Mask edge red
        edges = cv2.Canny((gray < 128).astype(np.uint8) * 255, 50, 150)
        canvas[edges > 0] = (0, 0, 255)
        # Cube green
        draw_cube(canvas, (0, 255, 0))

        # Predicted front line — cyan
        x_pred_m = fw_m + (y_pred[N - 1] / 100.0)
        pv_p = project_front_line(K, R_mat, t_m, x_pred_m, fw_m, fh_m, depth_m).astype(int)
        for j in range(len(pv_p)):
            p0 = tuple(pv_p[j])
            p1 = tuple(pv_p[(j + 1) % len(pv_p)])
            cv2.line(canvas, p0, p1, (255, 255, 0), args.line_width, cv2.LINE_AA)

        # Observed front line — yellow
        x_obs_m = fw_m + (y_obs[N - 1] / 100.0)
        pv_o = project_front_line(K, R_mat, t_m, x_obs_m, fw_m, fh_m, depth_m).astype(int)
        for j in range(len(pv_o)):
            p0 = tuple(pv_o[j])
            p1 = tuple(pv_o[(j + 1) % len(pv_o)])
            cv2.line(canvas, p0, p1, (0, 200, 200), args.line_width, cv2.LINE_AA)

        # Label
        cv2.putText(canvas, f"frame {N}  y_pred={y_pred[N-1]:.3f}cm  y_obs={y_obs[N-1]:.3f}cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out = cfg_dir / f"holdout_overlay_{N:02d}.png"
        cv2.imwrite(str(out), canvas)
        n_written += 1

    print(f"Wrote {n_written} holdout_overlay_NN.png to {cfg_dir}")
    print(f"  Green   = container cube wireframe")
    print(f"  Red     = actual binarized mask edge")
    print(f"  Cyan    = PREDICTED flow front (from joint θ̂)")
    print(f"  Yellow  = OBSERVED flow front (from y_obs)")


if __name__ == "__main__":
    main()

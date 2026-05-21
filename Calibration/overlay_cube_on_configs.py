"""Overlay the rendered cube wireframe (from current camera_params.xml) on top of
each config_NN.png to help visually identify container edges vs fluid extent.

Usage:
    python Calibration/overlay_cube_on_configs.py --data_dir <exp>

Output: <exp>/output/overlay_00.png ... overlay_08.png
        Each is the original config (grayscale) + green cube wireframe edges +
        red fluid mask edge from the binarized config.
"""
import argparse
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
    """Inverse of save_xml: q (w,x,y,z) -> R_mat used by render."""
    rot_our = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.vstack([rot_our[:, 0], -rot_our[:, 1], -rot_our[:, 2]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--line_width", type=int, default=2)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cfg_dir = data_dir / "output"

    cam = ET.parse(cfg_dir / "camera_params.xml").getroot().find("camera")
    eye_cm = np.array([float(v) for v in cam.attrib["eyepos"].split()])
    qw, qx, qy, qz = [float(v) for v in cam.attrib["quat"].split()]
    fov_deg = float(cam.attrib["fov"])

    R_mat = quat_to_R_mat(qw, qx, qy, qz)
    eye_m = eye_cm / 100.0
    t_m = -R_mat @ eye_m

    settings = _parse_settings_xml(str(data_dir / "settings.xml"))
    fw_m = settings["W"] / 100.0
    fh_m = settings["H"] / 100.0

    # Read first config to get image size + intrinsics
    cfg0 = cv2.imread(str(cfg_dir / "config_00.png"), cv2.IMREAD_GRAYSCALE)
    H_img, W_img = cfg0.shape
    f = H_img / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0, W_img/2.0], [0, f, H_img/2.0], [0, 0, 1.0]])

    # Project cube vertices once
    verts_3d = _cube_verts(fw_m, fh_m)
    pv = _project_KRt(verts_3d, K, R_mat, t_m)

    # Determine visible faces from eye
    visible_face_idx = []
    for fi, face in enumerate(_CUBE_FACES):
        center = verts_3d[face].mean(axis=0)
        if _FACE_NORMALS[fi] @ (eye_m - center) > 0:
            visible_face_idx.append(fi)

    def draw_cube(canvas, color):
        """Draw visible face edges on a BGR canvas in given color."""
        for fi in visible_face_idx:
            face = _CUBE_FACES[fi]
            corners = pv[face].astype(int)
            for j in range(len(corners)):
                p0 = tuple(corners[j])
                p1 = tuple(corners[(j + 1) % len(corners)])
                cv2.line(canvas, p0, p1, color, args.line_width, cv2.LINE_AA)
        return canvas

    # Generate overlay for each config
    n = 0
    for i in range(9):
        cfg_path = cfg_dir / f"config_{i:02d}.png"
        if not cfg_path.is_file():
            continue
        gray = cv2.imread(str(cfg_path), cv2.IMREAD_GRAYSCALE)
        # Convert to BGR for color overlay
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Red: fluid mask edge (so user sees what was binarized)
        edges = cv2.Canny((gray < 128).astype(np.uint8) * 255, 50, 150)
        canvas[edges > 0] = (0, 0, 255)  # red

        # Green: rendered cube wireframe
        draw_cube(canvas, (0, 255, 0))

        out = cfg_dir / f"overlay_{i:02d}.png"
        cv2.imwrite(str(out), canvas)
        n += 1

    print(f"Wrote {n} overlay_NN.png to {cfg_dir}")
    print(f"  Green = rendered cube wireframe (container edges)")
    print(f"  Red   = binarized fluid mask edge")
    print(f"  Gray  = original config background")


if __name__ == "__main__":
    main()

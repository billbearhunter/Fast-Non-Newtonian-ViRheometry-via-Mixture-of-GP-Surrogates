"""Render Background_mask + diff against config_00 from camera_params.xml.

Useful after manual tweaks (skips ChArUco re-fit).

Usage:
    python Calibration/render_from_xml.py --data_dir <exp>
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
    render_mask_KRt, render_background_KRt, diff_visual, diff_binary,
    _parse_settings_xml,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cfg_dir = data_dir / "output"
    xml_path = cfg_dir / "camera_params.xml"

    cam = ET.parse(xml_path).getroot().find("camera")
    eye_cm = np.array([float(v) for v in cam.attrib["eyepos"].split()])
    qw, qx, qy, qz = [float(v) for v in cam.attrib["quat"].split()]
    fov_deg = float(cam.attrib["fov"])

    # Inverse of save_xml: q_wxyz -> rot_our (cols = C_x, C_y, C_z)
    # then R_mat = [C_x; -C_y; -C_z]
    rot_obj = R.from_quat([qx, qy, qz, qw])  # scipy uses (x,y,z,w)
    rot_our = rot_obj.as_matrix()
    C_x = rot_our[:, 0]
    C_y = rot_our[:, 1]
    C_z = rot_our[:, 2]
    R_mat = np.vstack([C_x, -C_y, -C_z])
    eye_m = eye_cm / 100.0
    t_m = -R_mat @ eye_m

    target = cv2.imread(str(cfg_dir / "config_00.png"), cv2.IMREAD_GRAYSCALE)
    Himg, Wimg = target.shape

    # Inverse of save_xml fov: K[0,0] = H_px / (2 * tan(fov/2))
    f = Himg / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0, Wimg/2.0], [0, f, Himg/2.0], [0, 0, 1.0]])

    settings = _parse_settings_xml(str(data_dir / "settings.xml"))
    fw_m = settings["W"] / 100.0
    fh_m = settings["H"] / 100.0

    mask = render_mask_KRt(K, R_mat, t_m, fw_m, fh_m, Wimg, Himg)
    cv2.imwrite(str(cfg_dir / "Background_mask.png"), mask)
    cv2.imwrite(str(cfg_dir / "diff_combined.png"), diff_visual(mask, target))
    cv2.imwrite(str(cfg_dir / "diff_binary.png"),   diff_binary(mask, target))

    bg = sorted(data_dir.glob("*.JPG")) + sorted(data_dir.glob("*.jpg"))
    if bg:
        bg_img = cv2.imread(str(bg[0]))
        out = render_background_KRt(bg_img, K, R_mat, t_m, fw_m, fh_m,
                                    out_w=Wimg, out_h=Himg)
        cv2.imwrite(str(cfg_dir / "Background.png"), out)

    fg_m = mask < 128
    fg_t = target < 128
    iou = np.logical_and(fg_m, fg_t).sum() / max(np.logical_or(fg_m, fg_t).sum(), 1)
    print(f"IoU = {iou:.4f}")


if __name__ == "__main__":
    main()

"""Propose a random 1st setup (W, H) and create the ref folder + settings.xml.

Usage:
    python propose_initial_setup/propose_initial_setup.py -m Chuno -d 1.0

Outputs:
    data/real_world_experiments/ref_<material>_<H>_<W>_1/
    data/real_world_experiments/ref_<material>_<H>_<W>_1/settings.xml
"""
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

random.seed(datetime.now().timestamp())

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REF_ROOT = REPO / "data" / "new_real_world_experiments"
OUT_XML_SH = Path(__file__).resolve().parent / "out_xml.sh"


def write_settings_xml(out_dir: Path, root_dir_path: str,
                       H: float, W: float, rho: float) -> Path:
    """Write settings.xml directly (no bash required)."""
    Hs = str(round(H, 2))
    Ws = str(round(W, 2))
    xml = f"""<?xml version="1.0"?>
<Optimizer>
  <path
    root_dir_path="{root_dir_path}"
    GL_render_path="../libs/3D/GLRender3d/build/GLRender3d"
    mpm_path="../libs/3D/MPM3d/AGTaichiMPM.py"
    particle_skinner_path="../libs/ParticleSkinner3DTaichi.py"
    shell_script_dir_path="../libs/3D/shellScript3d"
    GL_emulation_render_path="../libs/3D/GLEmulationRender3d/build/GLEmulationRender3d"
  />

  <setup
    RHO="{rho}"
    H="{Hs}"
    W="{Ws}"
  />
  <cuboid min="-0.150000 -0.150000 -0.150000" max="{Ws} {Hs} 4.150000" density="{rho}" cell_samples_per_dim="2" vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100.000000 -1.000000 -100.000000" max="100.000000 0.000000 100.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 0.000000" max="0.000000 20.000000 4.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 -0.300000" max="{Ws} 20.000000 0.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 4.000000" max="{Ws} 20.000000 4.300000" boundary_behavior="sticking"/>

</Optimizer>
"""
    path = out_dir / "settings.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--density", type=float, required=True,
                        help="Material density ρ (g/cm³)")
    parser.add_argument("-m", "--material-name", required=True,
                        help="Material name (used in folder name)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REF_ROOT,
                        help="Root folder for new experiments (default: data/new_real_world_experiments)")
    parser.add_argument("--H", type=float, default=None,
                        help="Override H (cm). Default: random in [2.0, 7.0]")
    parser.add_argument("--W", type=float, default=None,
                        help="Override W (cm). Default: random in [2.0, 7.0]")
    args = parser.parse_args()

    ref_root = args.data_root
    ref_root.mkdir(parents=True, exist_ok=True)

    H = args.H if args.H is not None else round(random.randint(20, 70) * 0.1, 2)
    W = args.W if args.W is not None else round(random.randint(20, 70) * 0.1, 2)
    Hcm = str(round(H, 2))
    Wcm = str(round(W, 2))

    ref_name = f"ref_{args.material_name}_{Hcm}_{Wcm}_1"
    out_dir = ref_root / ref_name
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dir_path = str(ref_root / f"{args.material_name}_1")
    xml_path = write_settings_xml(out_dir, root_dir_path, H, W, args.density)

    print(f"Created : {out_dir}")
    print(f"Written : {xml_path}")
    print()
    print(f"Before running setup1.py, place the following into \"{out_dir}\":")
    print(f"  -  y_obs.npy          (8-frame flow distance processed from video)")
    print(f"  -  camera_params.xml  (from calibration)")
    print()
    print(f"Settings: material={args.material_name}, H={Hcm} cm, W={Wcm} cm, density={args.density}")
    print()
    print(f"Then run:")
    print(f"  python -m Optimization.setup1 \\")
    print(f"    --state-root Models/v10_yshape_v3p2_round2partial \\")
    print(f"    -f {out_dir} -d {args.density}")


if __name__ == "__main__":
    main()

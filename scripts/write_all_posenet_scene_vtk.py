import argparse
import os

from scripts.write_posenet_scene_vtk import main as write_one_main


def write_scene(scene_dir):
    import sys

    old_argv = sys.argv
    sys.argv = ["write_posenet_scene_vtk.py", "--scene-dir", scene_dir]
    write_one_main()
    sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/rigid_real")
    args = parser.parse_args()

    for dirpath, _, filenames in os.walk(args.root):
        if "scene_arrays.npz" in filenames:
            write_scene(dirpath)


if __name__ == "__main__":
    main()

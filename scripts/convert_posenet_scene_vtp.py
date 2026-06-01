import argparse
import os

from paraview.simple import LegacyVTKReader, SaveData


def convert(src, dst):
    data = LegacyVTKReader(FileNames=[src])
    SaveData(dst, proxy=data)
    print(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="outputs/posenet_scene")
    args = parser.parse_args()

    for name in ["mesh_before", "mesh_after", "point_cloud"]:
        convert(
            os.path.join(args.scene_dir, "{}.vtk".format(name)),
            os.path.join(args.scene_dir, "{}.vtp".format(name)),
        )


if __name__ == "__main__":
    main()

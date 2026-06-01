import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.estimate_posenet_rt import (
    load_config,
    load_posenet,
    load_target,
    make_batch,
    sample_surface,
    to_device,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("surface")
    parser.add_argument("point_cloud")
    parser.add_argument("--config", default="configs/test/main_config.yaml")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=0.001)
    parser.add_argument("--no-opencv-to-blender", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", default="outputs/posenet_scene")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    config = load_config(args.config, device)

    src = sample_surface(args.surface, args.points, args.source_scale)
    tgt = load_target(args.point_cloud, args.points, args.target_scale, not args.no_opencv_to_blender)
    batch = make_batch(src, tgt, args.surface, config, [64] * config.kpfcn_config.num_layers)
    batch = to_device(batch, device)

    model = load_posenet(config)
    with torch.no_grad():
        output = model(batch)

    R = output["R_s2t_pred"][0].detach().cpu().numpy()
    t = output["t_s2t_pred"][0].detach().cpu().numpy().reshape(3)

    mesh = o3d.io.read_triangle_mesh(args.surface)
    mesh_points = np.asarray(mesh.vertices, dtype=np.float32) * args.source_scale
    mesh_triangles = np.asarray(mesh.triangles, dtype=np.int32)
    mesh_points_aligned = mesh_points @ R.T + t

    np.savez(
        os.path.join(args.out_dir, "scene_arrays.npz"),
        mesh_before=mesh_points,
        mesh_after=mesh_points_aligned,
        faces=mesh_triangles,
        point_cloud=tgt,
    )

    with open(os.path.join(args.out_dir, "transform.json"), "w") as f:
        json.dump({"R": R.tolist(), "t": t.tolist()}, f, indent=2)

    print(os.path.join(args.out_dir, "transform.json"))


if __name__ == "__main__":
    main()

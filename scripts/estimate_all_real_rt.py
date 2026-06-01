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


def split_names(split):
    if split == "all":
        return ["train_real", "test_real"]
    return ["{}_real".format(split)]


def read_samples(data_root, patient, split):
    samples = []
    seen = set()
    for name in split_names(split):
        path = os.path.join(data_root, "{:02d}".format(patient), "{}.txt".format(name))
        with open(path, "r") as f:
            for line in f:
                sample = line.strip()
                if sample and sample not in seen:
                    samples.append(sample)
                    seen.add(sample)
    return samples


def patient_paths(data_root, patient):
    patient_dir = os.path.join(data_root, "{:02d}".format(patient))
    return patient_dir, os.path.join(patient_dir, "model", "reconstructed_mesh_world_m.obj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test/main_config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-root", default="outputs/rigid_real")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--patients", nargs="+", type=int, default=list(range(1, 22)))
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=0.001)
    parser.add_argument("--no-opencv-to-blender", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    config = load_config(args.config, device)
    data_root = args.data_root or config.data_root

    model = load_posenet(config)
    count = 0

    for patient in args.patients:
        patient_dir, mesh_path = patient_paths(data_root, patient)
        src = sample_surface(mesh_path, args.points, args.source_scale)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh_points = np.asarray(mesh.vertices, dtype=np.float32) * args.source_scale
        mesh_triangles = np.asarray(mesh.triangles, dtype=np.int32)

        for sample in read_samples(data_root, patient, args.split):
            point_cloud_path = os.path.join(patient_dir, "real", "liverPcds", "{}.ply".format(sample))
            tgt = load_target(
                point_cloud_path,
                args.points,
                args.target_scale,
                not args.no_opencv_to_blender,
            )
            batch = make_batch(src, tgt, mesh_path, config, [64] * config.kpfcn_config.num_layers)
            batch = to_device(batch, device)

            with torch.no_grad():
                output = model(batch)

            R = output["R_s2t_pred"][0].detach().cpu().numpy()
            t = output["t_s2t_pred"][0].detach().cpu().numpy().reshape(3)
            mesh_points_aligned = mesh_points @ R.T + t

            out_dir = os.path.join(args.out_root, args.split, "{:02d}".format(patient), sample)
            os.makedirs(out_dir, exist_ok=True)
            np.savez(
                os.path.join(out_dir, "scene_arrays.npz"),
                mesh_before=mesh_points,
                mesh_after=mesh_points_aligned,
                faces=mesh_triangles,
                point_cloud=tgt,
            )
            with open(os.path.join(out_dir, "transform.json"), "w") as f:
                json.dump({"R": R.tolist(), "t": t.tolist()}, f, indent=2)

            count += 1
            print("{:02d}/{} -> {}".format(patient, sample, out_dir))
            if args.max_samples is not None and count >= args.max_samples:
                return


if __name__ == "__main__":
    main()

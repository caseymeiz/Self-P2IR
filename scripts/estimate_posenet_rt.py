import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import torch
import yaml
from easydict import EasyDict as edict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from configs.models import architectures
from datasets.dataloader import collate_fn_3dmatch
from models.lepard.lepard import Pipeline


def load_config(path, device):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)
    config.device = device
    config.local_rank = 0
    config.gpus = 1
    config.kpfcn_config.architecture = architectures[config.dataset]
    return config


def sample_surface(path, n_points, scale):
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(path)
    else:
        pcd = mesh.sample_points_uniformly(n_points)
    points = np.asarray(pcd.points, dtype=np.float32) * scale
    if points.shape[0] == 0:
        raise ValueError("No source points could be read from {}".format(path))
    if points.shape[0] > n_points:
        points = points[np.random.default_rng(0).permutation(points.shape[0])[:n_points]]
    return points


def load_target(path, n_points, scale, opencv_to_blender):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.remove_duplicated_points()
    points = np.asarray(pcd.points, dtype=np.float32) * scale
    points = points[~np.all(points == 0.0, axis=1)]
    if points.shape[0] == 0:
        raise ValueError("No target points could be read from {}".format(path))
    if opencv_to_blender:
        points = np.dot(np.diag([1.0, -1.0, -1.0]), points.T).T
    if points.shape[0] > n_points:
        points = points[np.random.default_rng(0).permutation(points.shape[0])[:n_points]]
    return points.astype(np.float32)


def make_batch(src, tgt, mesh_path, config, neighborhood_limits):
    item = {
        "preope_pcd_src": src,
        "preope_pcd_src_normal": np.zeros_like(src, dtype=np.float32),
        "preope_reconstructed_faces": np.zeros((1, 3), dtype=np.int32),
        "intra_liver_pcd_tgt": tgt,
        "src_feats": np.ones((src.shape[0], 1), dtype=np.float32),
        "tgt_feats": np.ones((tgt.shape[0], 1), dtype=np.float32),
        "pose_R": np.eye(3, dtype=np.float32),
        "pose_t": np.zeros((3, 1), dtype=np.float32),
        "liver_labels": np.zeros((1, 1), dtype=np.int32),
        "liver_imgs": np.zeros((1, 1, 3), dtype=np.uint8),
        "correspondences": np.ones((1, 3), dtype=np.int32),
        "o3d_pre_mesh": mesh_path,
        "cam_K": np.eye(3, dtype=np.float32),
        "rgbs": np.zeros_like(tgt, dtype=np.float32),
        "img_size": np.array([1, 1], dtype=np.int32),
        "ocv2blender": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        "scale": np.array(1.0, dtype=np.float32),
        "bbx_center": np.zeros(3, dtype=np.float32),
        "which_patient": np.array(0, dtype=np.int32),
    }
    return collate_fn_3dmatch([item], config.kpfcn_config, neighborhood_limits)


def to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, list):
            if value and not isinstance(value[0], (str, np.ndarray, type(None))):
                batch[key] = [item.to(device) for item in value]
        elif not isinstance(value, (dict, float, np.ndarray, type(None))):
            batch[key] = value.to(device)
    return batch


def load_posenet(config):
    model = Pipeline(config).to(config.device)
    state = torch.load(config.pretrain, map_location="cpu")
    posenet_state = {
        key[len("posenet.") :]: value
        for key, value in state["state_dict"].items()
        if key.startswith("posenet.")
    }
    model.load_state_dict(posenet_state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("surface", help="Path to preoperative liver surface mesh or point cloud")
    parser.add_argument("point_cloud", help="Path to intraoperative point cloud")
    parser.add_argument("--config", default="configs/test/main_config.yaml")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=0.001)
    parser.add_argument("--no-opencv-to-blender", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

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
    print(json.dumps({"R": R.tolist(), "t": t.tolist()}, indent=2))


if __name__ == "__main__":
    main()

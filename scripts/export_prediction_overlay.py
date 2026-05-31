import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

from configs.models import architectures
from datasets.dataloader import get_dataloader, get_datasets
from lib.logger import setup_seed
from models.mynetwork import DeformRegis


def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["snapshot_dir"] = "snapshot/%s/%s" % (config["dataset"] + config["folder"], config["exp_dir"])
    config["tboard_dir"] = "snapshot/%s/%s/tensorboard" % (config["dataset"] + config["folder"], config["exp_dir"])
    config["save_dir"] = "snapshot/%s/%s/checkpoints" % (config["dataset"] + config["folder"], config["exp_dir"])
    config = edict(config)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() and config.gpu_mode else "cpu")
    config.local_rank = 0
    config.gpus = 1
    config.kpfcn_config.architecture = architectures[config.dataset]
    return config


def to_device(inputs, device):
    for k, v in inputs.items():
        if type(v) == list:
            if type(v[0]) not in [str, np.ndarray, None]:
                inputs[k] = [item.to(device) for item in v]
        elif type(v) not in [dict, float, type(None), np.ndarray]:
            inputs[k] = v.to(device)
    return inputs


def make_overlay(image, mask, alpha):
    image = image.astype(np.uint8)
    mask = mask > 0
    overlay = image.copy()
    overlay[mask] = (0, 220, 40)
    blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)

    contour_mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test/main_config.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/predicted_deformation_overlay.png")
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    setup_seed(0)
    torch.manual_seed(0)

    config = load_config(args.config)
    _, _, test_set = get_datasets(config)
    test_loader, neighborhood_limits = get_dataloader(test_set, config, mode="test", shuffle=False)
    test_loader, _ = get_dataloader(test_set, config, mode="test", shuffle=False, neighborhood_limits=neighborhood_limits)

    model = DeformRegis(config).to(config.device)
    state = torch.load(config.pretrain, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    loader_iter = iter(test_loader)
    inputs = None
    for _ in range(args.index + 1):
        inputs = next(loader_iter)
    inputs = to_device(inputs, config.device)

    data = model(inputs)

    image = inputs["ori_imgs"][0].detach().cpu().numpy()
    mask = data["mask"][0].detach().cpu().numpy()
    overlay = make_overlay(image, mask, args.alpha)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(args.output)


if __name__ == "__main__":
    main()

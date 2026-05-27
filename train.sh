#!/bin/bash
export PYTHONNOUSERSITE=1
export PYTHON_EGG_CACHE="/tmp/self-p2ir-python-eggs"
export CUDA_HOME="$PWD/.pixi/envs/default/pkgs/cuda-toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:$PWD/.pixi/envs/default/lib/python3.8/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PWD/models/pointnet/pointnet2:$PWD/extensions/earth_movers_distance:$PWD/extensions/chamfer_distance:$PWD/models/pointnet/pointnet2/dist/pointnet2-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/earth_movers_distance/dist/emd_cuda-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/chamfer_distance/dist/chamfer_3D-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH"
n_gpu=3 # number of gpu to use
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=$n_gpu  main.py --config="configs/train/main_config.yaml" --gpus=$n_gpu

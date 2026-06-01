#!/bin/bash
# Examples:
# pixi run bash estimate_all_real.sh
# pixi run bash estimate_all_real.sh --split test --patients 1 2 --max-samples 2

export PYTHONNOUSERSITE=1
export PYTHON_EGG_CACHE="/tmp/self-p2ir-python-eggs"
export CUDA_HOME="$PWD/.pixi/envs/default/pkgs/cuda-toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:$PWD/.pixi/envs/default/lib/python3.8/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PWD/models/pointnet/pointnet2:$PWD/extensions/earth_movers_distance:$PWD/extensions/chamfer_distance:$PWD/models/pointnet/pointnet2/dist/pointnet2-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/earth_movers_distance/dist/emd_cuda-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/chamfer_distance/dist/chamfer_3D-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH"

OUT_ROOT="${OUT_ROOT:-outputs/rigid_real}"
SPLIT="all"
PREV=""
for ARG in "$@"; do
    if [ "$PREV" = "--split" ]; then
        SPLIT="$ARG"
        break
    fi
    PREV="$ARG"
done

python scripts/estimate_all_real_rt.py --out-root "$OUT_ROOT" "$@"
if [ "$SPLIT" = "all" ]; then
    pixi run -e render python scripts/write_all_posenet_scene_vtk.py --root "$OUT_ROOT/all"
else
    pixi run -e render python scripts/write_all_posenet_scene_vtk.py --root "$OUT_ROOT/$SPLIT"
fi

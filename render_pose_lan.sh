#!/bin/bash
# Example:
# pixi run bash render_pose_lan.sh \
#   /shared/users/cjm6121/hybrid/Liver_regis/01/model/reconstructed_mesh_world_m.obj \
#   /shared/users/cjm6121/hybrid/Liver_regis/01/real/liverPcds/frame_7_json.ply

export PYTHONNOUSERSITE=1
export PYTHON_EGG_CACHE="/tmp/self-p2ir-python-eggs"
export CUDA_HOME="$PWD/.pixi/envs/default/pkgs/cuda-toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:$PWD/.pixi/envs/default/lib/python3.8/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PWD/models/pointnet/pointnet2:$PWD/extensions/earth_movers_distance:$PWD/extensions/chamfer_distance:$PWD/models/pointnet/pointnet2/dist/pointnet2-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/earth_movers_distance/dist/emd_cuda-0.0.0-py3.8-linux-x86_64.egg:$PWD/extensions/chamfer_distance/dist/chamfer_3D-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH"

OUT_DIR="${OUT_DIR:-outputs/posenet_scene}"
LAN_ID="${LAN_ID:-self-p2ir-pose-render}"

python scripts/export_posenet_scene_vtk.py "$1" "$2" --out-dir "$OUT_DIR" "${@:3}"
pixi run -e render python scripts/write_posenet_scene_vtk.py --scene-dir "$OUT_DIR"
pixi run -e render pvpython scripts/convert_posenet_scene_vtp.py --scene-dir "$OUT_DIR"
pixi run -e render pvpython --force-offscreen-rendering scripts/render_posenet_scene_paraview.py --mesh-path "$OUT_DIR/mesh_before.vtk" --points-path "$OUT_DIR/point_cloud.vtk" --png-path "$OUT_DIR/before.png"
pixi run -e render pvpython --force-offscreen-rendering scripts/render_posenet_scene_paraview.py --mesh-path "$OUT_DIR/mesh_after.vtk" --points-path "$OUT_DIR/point_cloud.vtk" --png-path "$OUT_DIR/after.png"
python scripts/stitch_pose_render.py --before "$OUT_DIR/before.png" --after "$OUT_DIR/after.png" --output "$OUT_DIR/before_after.png"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/mesh_before.vtp"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/mesh_after.vtp"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/point_cloud.vtp"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/mesh_before.vtk"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/mesh_after.vtk"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/point_cloud.vtk"
python /home/cjm6121/.codex/skills/codex-lan-viewer/scripts/show.py "$LAN_ID" "$OUT_DIR/before_after.png"

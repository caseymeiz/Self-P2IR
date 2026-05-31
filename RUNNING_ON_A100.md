# Running On This A100 Server

This checkout is set up to run with Pixi on the local A100 server.

## What Was Set Up

- Dataset path: `/shared/users/cjm6121/hybrid/Liver_regis`
- Pretrained checkpoint: `checkpoints/model_best_loss.pth`
- Test config: `configs/test/main_config.yaml`
- Pixi env: `pixi.toml` / `pixi.lock`

The bundled CUDA extension eggs did not run on A100 because they were compiled for a different GPU architecture. We installed CUDA toolkit 11.7 through Pixi and rebuilt these extensions for A100 `sm_80`:

- `models/pointnet/pointnet2/pointnet2_cuda...so`
- `extensions/earth_movers_distance/emd_cuda...so`
- `extensions/chamfer_distance/chamfer_3D...so`

`test.sh` and `train.sh` set `CUDA_HOME`, `LD_LIBRARY_PATH`, and `PYTHONPATH` so these rebuilt extensions are used before the old eggs.

## Run Test

```bash
pixi run bash test.sh
```

This loads `checkpoints/model_best_loss.pth` and evaluates on the real test split from `test_real.txt`.

## Model Stages

Self-P2IR uses the Lepard-style `posenet` module for the initial rigid registration. That part predicts the rigid transform `R_s2t_pred` and `t_s2t_pred`.

The downloaded checkpoint is the synthetic pretraining checkpoint for this rigid registration stage. Real-data fine-tuning loads that checkpoint, freezes `posenet`, and trains the later non-rigid/deformation path using the real training split from `train_real.txt`.

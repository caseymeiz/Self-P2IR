# AGENTS.md

This repository is a fork of Self-P2IR. We have been using it to reproduce the authors' provided pretrained-model results on the local A100 server.

For the server-specific setup, dataset path, checkpoint path, Pixi environment, rebuilt CUDA extensions, and run command, read:

- `RUNNING_ON_A100.md`

We also added helper code under `scripts/`, including prediction overlay and wireframe rendering utilities.

Do not assume the original bundled CUDA extension eggs work on this server; use the Pixi setup and rebuilt extensions documented in `RUNNING_ON_A100.md`.

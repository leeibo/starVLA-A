# oft_no_no_0_wos

This managed run trains pure `QwenOFT` on Astribot LeRobot data with:

- default base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- action head: MLP regression over OFT action query tokens
- history: disabled (`history.enabled: false`, `history.max_frames: 0`)
- VLM think branch: disabled
- state input: disabled (`include_state: false`)

In the run name, the first `no` means no VLM think supervision branch, the
second `no` means no history keyframes, `0` means zero history frames, and
`wos` means without state.

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/run_train.sh
```

Submit through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/submit_yhbatch.sh
```

The submit script defaults to `a800x`, 8 CPU cores, and the local `starVLA`
conda environment's `accelerate`. If `NUM_PROCESSES` is not set, it infers the
process count from the visible GPU count. It writes stdout/stderr copies to
`examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/logs/`.
It uses `/XYAIFS00/HDD_POOL/hlkj_zql/hlkj_zql_8/code/starVLA` as the default
repo root; override `STARGVLA_REPO_ROOT` only if the checkout moves.

Training parameters are read from `config.yaml`. The launch script only sets
launcher/runtime details such as `NUM_PROCESSES`, NCCL environment variables,
and the accelerate config path. It also falls back to
`conda run --no-capture-output -n starVLA accelerate` when `accelerate` is not
already on `PATH`.

W&B uses `wandb_project` and `wandb_entity` from `config.yaml`. The script
loads credentials from the ignored root `${REPO_ROOT}/.env` file:

```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_MODE=online
```

This run starts from the configured plain Qwen3-VL base checkpoint and does not
load a managed-run `trainer.pretrained_checkpoint` by default. `QwenOFT` does
not use FAST `<robot_action_*>` token supervision; actions are trained through
the MLP regression head.

Launch with a different process count:

```bash
NUM_PROCESSES=4 WANDB_MODE=disabled \
  bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/run_train.sh
```

For a multi-GPU batch job, request the GPU count from `yhbatch`; the submit
script should infer `NUM_PROCESSES` automatically. Set `NUM_PROCESSES` manually
only if the scheduler environment exposes the wrong visible GPU count.

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/run_policy_server.sh
```

The server auto-selects the latest checkpoint under
`results/Checkpoints/oft_no_no_0_wos` unless `POLICY_CKPT_PATH` is set. Default
port is `7980`.

Outputs are written to `results/Checkpoints/oft_no_no_0_wos`.

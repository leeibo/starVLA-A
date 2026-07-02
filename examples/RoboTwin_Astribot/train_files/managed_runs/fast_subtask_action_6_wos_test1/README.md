# fast_subtask_action_6_wos_test1

This managed run trains `QwenFast` on only the
`beat_block_hammer_rotate_view` Astribot LeRobot task with:

- default base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`
- data mix: `robotwin_astribot_task1`
- history mode: `action_keyframe`
- think supervision: `subtask_instruction`
- history length: 6 history frames (`history.max_frames: 6`)
- state input: disabled (`include_state: false`)
- managed-run initialization checkpoint: none
- resume mode: disabled (`trainer.is_resume: false`)

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_train.sh
```

Submit through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/submit_yhbatch.sh
```

The submit script defaults to `a800x`, 8 CPU cores, and the local `starVLA`
conda environment's `accelerate`. If `NUM_PROCESSES` is not set, it infers the
process count from the visible GPU count. It writes stdout/stderr copies to
`examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/logs/`.
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

The VLM path must point to an offline `*-Action` checkpoint that already
contains the 2048 FAST action tokens. This run keeps
`framework.action_model.auto_add_action_tokens: false`.

This run does not set `trainer.pretrained_checkpoint`, so it does not load
weights from `fast_subtask_action_12_wos`. With `trainer.is_resume: false`,
training starts from the configured base VLM checkpoint with a fresh optimizer
and scheduler.

Launch with a different process count:

```bash
NUM_PROCESSES=4 WANDB_MODE=disabled \
  bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_train.sh
```

For a multi-GPU batch job, request the GPU count from `yhbatch`; the submit
script should infer `NUM_PROCESSES` automatically. Set `NUM_PROCESSES` manually
only if the scheduler environment exposes the wrong visible GPU count.

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_policy_server.sh
```

The server auto-selects the latest checkpoint under
`results/Checkpoints/fast_subtask_action_6_wos_test1` unless `POLICY_CKPT_PATH` is
set. Default port is `7980`.

Outputs are written to `results/Checkpoints/fast_subtask_action_6_wos_test1`.

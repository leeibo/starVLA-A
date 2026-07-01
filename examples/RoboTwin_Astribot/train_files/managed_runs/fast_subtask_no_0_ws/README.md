# fast_subtask_no_0_ws

This managed run trains `QwenFastState` on Astribot LeRobot data with:

- default base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`
- history: disabled (`history.enabled: false`, `history.max_frames: 0`)
- think supervision: `subtask_instruction`
- state input: enabled (`include_state: true`)
- state encoder: one current-frame state token, 18-D state projected to the VLM hidden size

In the run name, `no` is the history-keyframe field and means no history
frames are used. It is not written as `history.mode: no`; the dataloader only
uses `history.mode` when history is enabled.

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_train.sh
```

Submit through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/submit_yhbatch.sh
```

The submit script defaults to `a800x`, 8 CPU cores, and the local `starVLA`
conda environment's `accelerate`. If `NUM_PROCESSES` is not set, it infers the
process count from the visible GPU count. It writes stdout/stderr copies to
`examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/logs/`.
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

This run starts from the offline `Qwen3-VL-2B-Instruct-Action` base checkpoint.
It does not load a `QwenFast` / `wos` training checkpoint, because
`QwenFastState` adds a state encoder whose parameters are not present in those
checkpoints.

Launch with a different process count:

```bash
NUM_PROCESSES=4 WANDB_MODE=disabled \
  bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_train.sh
```

For a multi-GPU batch job, request the GPU count from `yhbatch`; the submit
script should infer `NUM_PROCESSES` automatically. Set `NUM_PROCESSES` manually
only if the scheduler environment exposes the wrong visible GPU count.

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_policy_server.sh
```

The server auto-selects the latest checkpoint under
`results/Checkpoints/fast_subtask_no_0_ws` unless `POLICY_CKPT_PATH` is set.
Default port is `7980`. Clients must send `state_history` or `state` for this
`ws` run; for no-history inference this is the current-frame 18-D state.

Outputs are written to `results/Checkpoints/fast_subtask_no_0_ws`.

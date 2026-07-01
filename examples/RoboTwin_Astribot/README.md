# RoboTwin Astribot

This example namespace contains the Astribot-specific RoboTwin training setup.
It is intentionally separate from `examples/Robotwin`, which keeps the original
RobotWin Agilex/ARX benchmark files.

Astribot-specific contents:

- `train_files/data_registry/data_config.py`: head-camera-only Astribot registry.
- `train_files/starvla_fast_robotwin_astribot_history.yaml`: QwenFast history config.
- `train_files/starvla_fast_state_robotwin_astribot_history.yaml`: QwenFastState history config.
- `train_files/managed_runs/fast_subtask_action_12_wos/`: managed 12-history-frame training run.
- `train_files/managed_runs/fast_subtask_action_6_wos/`: managed 6-history-frame run initialized from the 12-frame checkpoint.
- `train_files/managed_runs/fast_subtask_action_6_ws/`: managed 6-history-frame QwenFastState run with state tokens.
- `train_files/managed_runs/fast_subtask_no_0_wos/`: managed 0-history-frame QwenFast run.
- `train_files/managed_runs/fast_subtask_no_0_ws/`: managed 0-history-frame QwenFastState run with state tokens.

## 4-GPU training notes

The managed run `fast_subtask_action_12_wos` was smoke-tested on a Slurm
node with 4 H100 GPUs. `fast_subtask_action_6_wos` uses the same launcher
pattern and initializes from the 12-frame checkpoint. `fast_subtask_action_6_ws`
uses `QwenFastState` and conditions on one state token per input frame. Use the
`starVLA` conda environment. `fast_subtask_no_0_wos` disables history entirely
and uses only the current image. `fast_subtask_no_0_ws` is the matching
no-history state-token run.

The default base VLM is:

```bash
./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action
```

This is an offline action-token checkpoint generated from plain
`Qwen3-VL-2B-Instruct` by adding the 2048 `<robot_action_*>` FAST tokens. The
training configs keep `framework.action_model.auto_add_action_tokens: false`,
so startup fails clearly if a plain checkpoint is used by mistake.

Both the model and datasets may be symlinks under `playground/`; use `ls -la`
or `find -L` when checking them.

Full training:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_train.sh
```

Queue the same run through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/submit_yhbatch.sh
```

Start its inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_policy_server.sh
```

6-frame continuation from the 12-frame checkpoint:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/run_train.sh
```

Queue the 6-frame run through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/submit_yhbatch.sh
```

Start its inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/run_policy_server.sh
```

6-frame state-token run:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/run_train.sh
```

Queue the 6-frame state-token run through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/submit_yhbatch.sh
```

Start its inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/run_policy_server.sh
```

0-history no-state run:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/run_train.sh
```

Queue the 0-history run through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/submit_yhbatch.sh
```

Start its inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/run_policy_server.sh
```

0-history state-token run:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_train.sh
```

Queue the 0-history state-token run through `yhbatch`:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/submit_yhbatch.sh
```

Start its inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_policy_server.sh
```

Smoke test:

```bash
NUM_PROCESSES=4 WANDB_MODE=disabled \
  bash examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_train.sh
```

Expected successful signals:

- `Distributed environment: DEEPSPEED Backend: nccl`
- `Num processes: 4`
- `Total batch size = 16`
- `Step 1, Loss: ...`
- GPUs are released after exit.

Operational notes from the first run:

- DeepSpeed import needs `CUDA_HOME`; on the tested node, CUDA 12.4 lives at
  `/APP/u22/ai_x86/CUDA/12.4`. The managed run script now sets this
  automatically when `CUDA_HOME` is unset and the path exists.
- The original `NCCL_SOCKET_IFNAME=bond0` failed on the tested node because
  there is no `bond0`; the available interface was `ib0`. The run script now
  chooses `bond0`, then `ib0`, then `eth0`.
- The FAST action tokenizer is stored locally at
  `./playground/Pretrained_models/fast` as real files, not a symlink. The code
  defaults to this path and falls back to `physical-intelligence/fast` only if
  the local directory is missing.
- Full `data_mix: robotwin_astribot` scans all Astribot tasks and can spend a
  long time in dataset statistics / indexing before the first optimizer step.
  Use a temporary config with `data_mix: robotwin_astribot_task1` for fast
  connectivity checks.
- For smoke tests that change training steps or data mix, edit the managed
  run's `config.yaml`; `run_train.sh` intentionally does not override training
  semantics.
- For queued jobs, use the run-local `submit_yhbatch.sh`. It writes a local
  stdout/stderr copy under that run's `logs/` directory and defaults
  `NUM_PROCESSES` to the requested visible GPU count. Override
  `NUM_PROCESSES` manually only if the scheduler environment exposes the wrong
  GPU count.
- For inference, use the run-local `run_policy_server.sh`. The server defaults
  to port `7980` and auto-selects the latest checkpoint under
  `results/Checkpoints/<run_id>`; set `POLICY_CKPT_PATH` to pin a specific
  checkpoint.

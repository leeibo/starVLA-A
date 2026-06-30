# Change Log

This directory tracks implementation changes made during iterative development.
Append a new entry for every meaningful code/config/doc change.

## Entry Template

```markdown
## YYYY-MM-DD - Short Title

Scope:
- File or module touched

Changes:
- Concrete behavior/config/code change

Validation:
- Command or manual check performed

Notes:
- Remaining risk, assumptions, or follow-up
```

## 2026-06-28 - FAST Multi-Frame History Prompt Support

Scope:
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `starVLA/model/modules/vlm/QWen2_5.py`
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/model/modules/vlm/QWen3_5.py`
- `starVLA/model/modules/vlm/chat_label_utils.py`
- `examples/Robotwin/train_files/data_registry/data_config.py`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`

Changes:
- Added configurable LeRobot history-frame sampling through `datasets.vla_data.history`.
- Added `action_keyframe`, `motion_keyframe`, and `subtask_keyframe` history modes.
- Added head-camera-only Robotwin/Astribot robot types using `video.camera_head`.
- Added FAST answer templating through `datasets.vla_data.fast_answer.template`.
- Added configurable supervised instruction source through `datasets.vla_data.fast_answer.instruction_source`.
- Changed Qwen VLM label masking so the full assistant answer is supervised, including `<think>` text and `<action>` FAST tokens.
- Added an example FAST training config for `playground/dataset/RoboTwin_Astribot`.

Validation:
- `conda run -n starVLA python -m py_compile ...`
- YAML parse check for `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`.
- Registry check for `robotwin_astribot`.
- Lightweight history-index checks for all three history modes.
- `git diff --check`.

Notes:
- `history.max_frames: null` means no limit on extra history frames.
- Numeric `history.max_frames` means maximum extra history frames; total images are at most `max_frames + 1` including current frame.
- `fast_answer.instruction_source` accepts `instruction`, `subtask_instruction`, or `auto`; `instruction_key` can point to a custom sample field.
- `AGENT.md` was already untracked and was not modified.

## 2026-06-28 - Astribot LeRobot Data Link and Loader Compatibility

Scope:
- `playground/dataset/RoboTwin_Astribot_lerobot`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `examples/Robotwin/train_files/data_registry/data_config.py`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`

Changes:
- Added a separate symlink `playground/dataset/RoboTwin_Astribot_lerobot` to the converted LeRobot Astribot root, keeping the original raw-data symlink unchanged.
- Updated the FAST Astribot YAML to use the LeRobot symlink and `action_dim: 18`.
- Changed Astribot data config to use head camera only plus `state.astribot` and `action.astribot` 18-D fields.
- Replaced Astribot mixtures with the task directory names that actually exist in the converted LeRobot root.
- Added a fallback that infers LeRobot modality metadata from `meta/info.json` when `meta/modality.json` is absent.
- Added Astribot subtask instruction lookup through `meta/astribot_subtask_metadata.json` and parquet `subtask_instruction_index`.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/dataloader/gr00t_lerobot/datasets.py examples/Robotwin/train_files/data_registry/data_config.py`
- Parsed `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`.
- Instantiated `shake_bottle_rotate_view` from `playground/dataset/RoboTwin_Astribot_lerobot` and extracted step 150.
- Saved sample contact sheet to `/tmp/starvla_astribot_lerobot_sample_step150.jpg`.

Notes:
- The extracted sample has 10 images, history indices `[6, 22, 38, 54, 70, 86, 102, 118, 134, 150]`, and action shape `(16, 18)`.
- The supervised subtask text resolves to `find bottle, grasp it, and shake it`.

## 2026-06-28 - QwenFastState State-History Conditioning

Scope:
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/model/framework/VLM4A/QwenFastState.py`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`

Changes:
- Added `state_history` packing when `datasets.vla_data.include_state: true`; it uses the same frame indices as history images.
- Added `QwenFastState`, a Qwen3-only FAST framework that projects each low-dimensional state vector to one soft token.
- Inserted state soft tokens before the supervised assistant response while masking their labels with `-100`.
- Added an Astribot FAST-State training config with `state_dim: 18` and `include_state: true`.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/dataloader/gr00t_lerobot/datasets.py starVLA/model/framework/VLM4A/QwenFastState.py`
- Parsed `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`.
- Extracted `shake_bottle_rotate_view` step 150 and verified `state_history` shape `(10, 18)` matches 10 input images.
- Verified `QwenFastState` registration through `FRAMEWORK_REGISTRY`.

Notes:
- The FAST action target remains `<robot_action_x>` next-token prediction; state is conditioning context only.
- QwenFastState currently supports Qwen3-VL only.

## 2026-06-28 - Managed FAST Subtask Action Training Run

Scope:
- `doc/training_run_naming.md`
- `AGENT.md`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/README.md`

Changes:
- Added the managed run directory for `fast_subtask_action_ultra_wos`.
- Added a QwenFast training config using action-keyframe history, subtask `<think>` supervision, unlimited history, and no state input.
- Added a launch script that binds `run_id` and output directory to the run name and supports environment-variable overrides.
- Documented the training-run naming convention and added an `AGENT.md` reminder to consult it for future training management.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`

## 2026-06-28 - Best-Effort FAST Action Eval

Scope:
- `starVLA/training/train_starvla.py`

Changes:
- Made periodic training-time action eval tolerant to the specific QwenFast failure where free generation produces no `<robot_action_*>` tokens.
- On this failure, training records `eval/action_decode_failed=1.0`, logs one warning on the main process, synchronizes ranks, and continues.
- Successful eval now records `eval/action_decode_failed=0.0` alongside `mse_score`.
- Other `RuntimeError` failures are still raised.

Reason:
- Offline `*-Action` checkpoints only add the tokenizer/embedding rows. Early in training, the model may not yet emit action tokens during free-generation eval, and that should not kill the main training loop.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/training/train_starvla.py`

## 2026-06-28 - Qwen3 Flash Attention Config Fix

Scope:
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/model/modules/vlm/QWen3_5.py`

Changes:
- Removed the hard-coded `attn_implementation = "sdpa"` override from Qwen3/Qwen3.5 VLM interfaces.
- `framework.qwenvl.attn_implementation: flash_attention_2` now takes effect when `flash_attn` is installed.
- Kept automatic fallback to `sdpa` when `flash_attention_2` is requested but no flash-attention backend is available.
- Added startup logging of the final attention backend used to load the model.

Validation:
- Confirmed the `starVLA` environment can find `flash_attn`.
- Confirmed the managed run config requests `flash_attention_2`.
- `conda run -n starVLA python -m py_compile starVLA/model/modules/vlm/QWen3.py starVLA/model/modules/vlm/QWen3_5.py`

## 2026-06-28 - Astribot Zero-Weight Task Filter

Scope:
- `examples/RoboTwin_Astribot/train_files/data_registry/data_config.py`
- `starVLA/dataloader/lerobot_datasets.py`

Changes:
- Added a `DISABLED_ASTRIBOT_TASKS` hook in the Astribot registry. The current set is empty, so all 35 tasks, including `rank_backside_rgb_blocks`, remain active.
- Made `get_vla_dataset` skip mixture entries with `weight <= 0.0` before dataset construction, so future disabled tasks are not initialized or sampled.

Validation:
- Verified `rank_backside_rgb_blocks` has mixture weight `1.0`.
- Built the Astribot mixture metadata and verified the active dataset list has 35 tasks and contains `rank_backside_rgb_blocks`.
- `conda run -n starVLA python -m py_compile examples/RoboTwin_Astribot/train_files/data_registry/data_config.py starVLA/dataloader/lerobot_datasets.py`

## 2026-06-28 - Rename Managed Run to 12-Frame Memory

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Renamed the managed run from `fast_subtask_action_ultra_wos` to `fast_subtask_action_12_wos`.
- Updated `config.yaml` run id to `fast_subtask_action_12_wos`.
- Set `datasets.vla_data.history.max_frames: 12`, meaning at most 12 history frames plus the current frame.
- Updated managed-run launch paths and README text.

Validation:
- Parsed `config.yaml` with `OmegaConf` and verified `run_id=fast_subtask_action_12_wos` and `history.max_frames=12`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_train.sh`

## 2026-06-28 - FAST Action Eval Skip Before Token Emission

Scope:
- `starVLA/training/train_starvla.py`

Changes:
- Made periodic training-time action eval best-effort for QwenFast/QwenFastState.
- If `predict_action()` generation produces no `<robot_action_*>` tokens, training now records `eval/action_decode_failed=1.0`, logs one warning, and continues.
- Kept `predict_action()` strict for actual inference/deployment, where empty action-token generation should still be treated as an error.

Reason:
- Offline `Qwen3-VL-2B-Instruct-Action` only expands the tokenizer/embedding table. It has not been fine-tuned to emit FAST action tokens yet, so early `generate()` eval can legitimately produce no action tokens.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/training/train_starvla.py`
- Parsed `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/config.yaml` with `OmegaConf`.
- Verified `run_id=fast_subtask_action_ultra_wos`, `framework=QwenFast`, `history.mode=action_keyframe`, `history.max_frames=None`, `include_state=False`, and `fast_answer.instruction_source=subtask_instruction`.
- Verified `OmegaConf.from_dotlist` parses CLI `history.max_frames null` as Python `None`.
- `git diff --check`.

Notes:
- `ultra` maps to `history.max_frames: null`.
- `wos` maps to `datasets.vla_data.include_state: false`.

## 2026-06-28 - RoboTwin Astribot Example Namespace

Scope:
- `examples/RoboTwin_Astribot/README.md`
- `examples/RoboTwin_Astribot/train_files/data_registry/data_config.py`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/`
- `examples/Robotwin/train_files/data_registry/data_config.py`
- `doc/training_run_naming.md`
- `AGENT.md`

Changes:
- Moved Astribot-specific training configs and managed runs out of the original `examples/Robotwin` namespace.
- Added `examples/RoboTwin_Astribot` as the dedicated example namespace for the converted Astribot LeRobot data.
- Moved the Astribot head-camera-only data registry into `examples/RoboTwin_Astribot/train_files/data_registry/data_config.py`.
- Restored the original Robotwin registry so it no longer contains Astribot robot types or mixtures.
- Updated training-run naming docs and `AGENT.md` to point future Astribot managed runs at `examples/RoboTwin_Astribot`.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- Parsed Astribot FAST, FASTState, and managed-run YAML files with `OmegaConf`.
- Verified the global LeRobot registry auto-discovers `robotwin_astribot` from `examples/RoboTwin_Astribot/train_files/data_registry/data_config.py`.
- Verified `DATASET_NAMED_MIXTURES["robotwin_astribot"]` contains 35 Astribot tasks.
- Verified `examples/Robotwin/train_files/data_registry/data_config.py` has no remaining diff from the tracked original file.
- `git diff --check`.

Notes:
- The global LeRobot registry auto-discovers `examples/*/train_files/data_registry`, so the new Astribot registry remains available without changing the dataloader entrypoint.

## 2026-06-28 - Tightened RoboTwin Astribot FAST Configs

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`

Changes:
- Removed unused `datasets.vlm_data` blocks from Astribot FAST configs.
- Removed unused cotrain/template leftovers such as `action_type`, `load_all_data_for_training`, `loss_scale`, `max_grad_norm`, trainer-level `weight_decay`, `gradient_accumulation_steps`, and optimizer `name`.
- Removed keyframe-column settings that are not used by `history.mode: action_keyframe`.
- Simplified the managed run script so fixed run semantics come from `config.yaml`; the script now keeps only launch-time overrides.

Validation:
- Parsed all three Astribot YAML files with `OmegaConf`.
- Verified no `datasets.vlm_data` remains in the Astribot FAST configs.
- Verified managed `QwenFast` remains `include_state=False`, `history.mode=action_keyframe`, and `history.max_frames=None`.
- Verified `QwenFastState` config remains `include_state=True`, `history.mode=action_keyframe`, and `history.max_frames=None`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- `git diff --check`.

## 2026-06-28 - Removed Managed Run Config Overrides

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/README.md`

Changes:
- Removed all trainer/model/dataset dotlist overrides from `run_train.sh`.
- Removed script-side output-directory construction so `run_id` and `run_root_dir` come only from `config.yaml`.
- Kept only launcher/runtime controls in the script: accelerate config, process count, conda fallback, and NCCL/runtime environment.
- Updated the managed-run README to state that training parameters should be changed in `config.yaml`.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- Verified the script only passes `--config_file`, `--num_processes`, and `--config_yaml`; no model/dataset/trainer dotlist overrides remain.
- Parsed `config.yaml` with `OmegaConf` and verified `datasets.vla_data.per_device_batch_size == 2`.
- `git diff --check`.

Notes:
- `config.yaml` is now the single source of truth for `per_device_batch_size`, `base_vlm`, `data_mix`, trainer schedule, `run_id`, W&B config, and related training semantics.

## 2026-06-29 - Managed Run fast_subtask_action_6_wos

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added a managed run derived from `fast_subtask_action_12_wos`.
- Set `run_id: fast_subtask_action_6_wos`.
- Set `datasets.vla_data.history.max_frames: 6`.
- Set `trainer.pretrained_checkpoint` to `results/Checkpoints/fast_subtask_action_12_wos/checkpoints/steps_100000_pytorch_model.pt`.
- Kept `trainer.is_resume: false`, so model weights load from the 12-frame checkpoint while optimizer and scheduler start fresh.

Validation:
- Confirmed the source checkpoint exists.
- Parsed the new `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/run_train.sh`
- `git diff --check`.

## 2026-06-29 - Astribot Managed Run yhbatch Scripts

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/submit_yhbatch.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/submit_yhbatch.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/run_train.sh`
- `examples/RoboTwin_Astribot/README.md`
- `doc/training_config_guidelines.md`

Changes:
- Added run-local `yhbatch` submission scripts for the active Astribot managed runs.
- Each submit script defaults to `a800x`, 8 CPU cores, one GPU, and the local `starVLA` conda environment's `accelerate`.
- Added run-local stdout/stderr capture under each run's `logs/` directory.
- Added automatic `NUM_PROCESSES` inference from scheduler GPU count, `CUDA_VISIBLE_DEVICES`, or `nvidia-smi -L`.
- Changed submit scripts to resolve the managed run directory from a fixed repo root instead of `BASH_SOURCE[0]`, because batch schedulers may execute a copied script from `/tmp/slurmd/...`.
- Made `run_train.sh` preserve externally provided `WANDB_MODE` instead of forcing offline mode.
- Made externally supplied `WANDB_MODE` take priority over root `.env`, so batch jobs can force `offline` or `disabled` even if `.env` is configured for online W&B.
- Documented the separation between training semantics in `config.yaml`, launch mechanics in `run_train.sh`, and queued job handling in `submit_yhbatch.sh`.

Validation:
- `bash -n` for both `submit_yhbatch.sh` files.
- `bash -n` for both updated `run_train.sh` files.
- `git diff --check`.

## 2026-06-30 - Managed Run Creation Checklist Requires yhbatch

Scope:
- `doc/training_run_naming.md`
- `doc/training_config_guidelines.md`
- `AGENT.md`

Changes:
- Updated the managed run creation rules so every formal training project includes `submit_yhbatch.sh` alongside `config.yaml`, `run_train.sh`, and `README.md`.
- Added a new managed-run checklist covering run directory creation, config naming, launch script validation, submit script validation, README updates, and changelog updates.
- Documented the minimum `submit_yhbatch.sh` requirements: fixed repo root with `STARGVLA_REPO_ROOT` override, `starVLA` accelerate binding, automatic `NUM_PROCESSES` inference, W&B mode override support, run-local logs, and calling `run_train.sh`.
- Updated `AGENT.md` so future training-script tasks explicitly consult the submit-script guidance.

Validation:
- `git diff --check`.

## 2026-06-30 - Managed Run fast_subtask_action_6_ws

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/`
- `examples/RoboTwin_Astribot/README.md`
- `doc/training_run_naming.md`

Changes:
- Added a managed run for `fast_subtask_action_6_ws`.
- Set `framework.name: QwenFastState` and `datasets.vla_data.include_state: true`.
- Added `framework.state_model` with 18-D state input, one state token per input frame, frame embeddings enabled, and `max_frames: 256`.
- Kept subtask `<think>` supervision, `action_keyframe` history, `history.max_frames: 6`, and the offline `Qwen3-VL-2B-Instruct-Action` base VLM.
- Added `trainer.learning_rate.state_encoder: 1.0e-04`.
- Added `run_train.sh`, `submit_yhbatch.sh`, and run-local README using the managed run conventions.
- Removed copied runtime logs from the new run directory.
- Updated the naming document so `fast` denotes the Qwen FAST-token training family and `ws/wos` selects `QwenFastState` vs `QwenFast`.

Validation:
- Parsed `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/submit_yhbatch.sh`
- `git diff --check`.

## 2026-06-30 - Astribot Managed Run Policy Server Scripts

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/run_policy_server.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/run_policy_server.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/run_policy_server.sh`
- `examples/RoboTwin_Astribot/README.md`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_wos/README.md`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos/README.md`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_ws/README.md`
- `doc/training_run_naming.md`
- `doc/training_config_guidelines.md`
- `AGENT.md`

Changes:
- Added run-local `run_policy_server.sh` scripts for the active Astribot managed runs.
- Each policy server script defaults to the local `starVLA` Python, port `7980`, bf16 enabled, and automatic latest-checkpoint selection under `results/Checkpoints/<run_id>`.
- `POLICY_CKPT_PATH` can pin a specific checkpoint.
- Documented that policy server scripts are required for managed runs, but `submit_policy_server_yhbatch.sh` should not be generated by default.
- Removed the accidentally generated `fast_subtask_no_0_wos` run directory; it will be created later from an explicit example request.

Validation:
- `bash -n` for the three added `run_policy_server.sh` scripts.
- Confirmed no `submit_policy_server_yhbatch.sh` files exist under the Astribot managed runs.
- Confirmed `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos` does not exist.
- `git diff --check`.

## 2026-06-28 - Training Config Writing Guidelines

Scope:
- `doc/training_config_guidelines.md`
- `doc/training_run_naming.md`
- `AGENT.md`

Changes:
- Added a dedicated guide for writing compact training configs and keeping `run_train.sh` free of training-semantics overrides.
- Simplified `AGENT.md` into a short document index for training naming and config-writing tasks.
- Updated the run naming doc to reference the config-writing guide and remove stale script override/output-copying language.

Validation:
- Verified `doc/training_config_guidelines.md` is not ignored by `.gitignore`.
- Checked `doc/training_run_naming.md` no longer documents stale `BASE_VLM` or script-side output-copying behavior.
- `git diff --check`.

Notes:
- Future training config edits should check `doc/training_config_guidelines.md` before changing YAML or launch scripts.

## 2026-06-28 - QwenFast Auto Action Token Addition

Scope:
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/config.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/README.md`
- `doc/training_config_guidelines.md`

Changes:
- Changed QwenFast action-token extraction to use the active tokenizer's `<robot_action_*>` ids instead of hard-coded interface private attributes.
- Added automatic `<robot_action_*>` special-token insertion for plain Qwen checkpoints when `framework.action_model.auto_add_action_tokens: true`.
- Added action-token embedding/lm-head row initialization controlled by `framework.action_model.action_token_init_strategy`.
- Documented that plain `Qwen3-VL-2B-Instruct` can be used as the FAST starting checkpoint because QwenFast now adds FAST action tokens at startup.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/model/framework/VLM4A/QwenFast.py starVLA/model/framework/VLM4A/QwenFastState.py`
- Ran a fake tokenizer/model test that starts without action tokens, auto-adds them, preserves existing special tokens, and decodes generated VLM token ids back to FAST ids.
- Parsed all RoboTwin Astribot FAST YAML configs and verified `auto_add_action_tokens=true`.

Notes:
- Existing `*-Action` checkpoints still work; QwenFast only adds missing action tokens.
- Plain Qwen checkpoints will train the newly added action-token rows from scratch.
- Superseded by the offline checkpoint convention below for managed Astribot FAST training.

## 2026-06-28 - Offline Qwen3-VL-2B Action Checkpoint

Scope:
- `starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py`
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/config.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/train_files/starvla_fast_state_robotwin_astribot_history.yaml`
- `examples/RoboTwin_Astribot/README.md`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/README.md`
- `doc/training_config_guidelines.md`
- `AGENT.md`

Changes:
- Generated `/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/ckpt/Qwen3-VL-2B-Instruct-Action` from plain `Qwen3-VL-2B-Instruct` by adding the 2048 FAST `<robot_action_*>` tokens offline.
- Added the project symlink `playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`.
- Updated Astribot FAST training configs to use `./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`.
- Set `framework.action_model.auto_add_action_tokens: false` in Astribot FAST configs so training no longer mutates the tokenizer/model online.
- Changed QwenFast's default `auto_add_action_tokens` value to `false`; the auto-add path remains available only when explicitly enabled for development.
- Hardened the offline token-add script for Qwen3 reserved embedding rows and made debugpy opt-in via `DEBUGPY_ENABLE=1`.
- Updated docs to make offline `*-Action` checkpoints the default FAST training convention.

Validation:
- Offline conversion reported action token id range `[151669, 153716]` and model embedding size `153717`.
- Reload check passed for all generated action tokens.
- Parsed/inspected Astribot FAST configs and verified they point to `Qwen3-VL-2B-Instruct-Action` with `auto_add_action_tokens: false`.

## 2026-06-28 - Managed Run W&B Secret Loading

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/README.md`
- `doc/training_config_guidelines.md`

Changes:
- Kept W&B project/entity in `config.yaml`.
- Added ignored root `.env` loading to the managed run script.
- Documented that `WANDB_API_KEY` belongs in local `.env` or the process environment, not in versioned scripts or docs.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_ultra_wos/run_train.sh`

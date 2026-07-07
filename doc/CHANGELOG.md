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

## 2026-07-06 - GR00T Dual-Head No-State Action Run

Scope:
- `starVLA/model/framework/VLM4A/QwenGR00TDual.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_wos/`

Changes:
- Added `QwenGR00TDual`, a no-state `QwenGR00T` variant with two independent GR00T DiT action heads.
- The arm head supervises action dimensions `[0:16]`; the camera head supervises action dimensions `[16:18]`; inference concatenates both predictions back to 18 dimensions.
- Added `gr00tdual_subtask_action_12_wos`, a cotrain managed run with action-keyframe history, 12 history frames, subtask think supervision, and state disabled.

Validation:
- Pending local syntax/config validation.

## 2026-07-06 - GR00T State Dual-Head Action Run

Scope:
- `starVLA/model/framework/VLM4A/QwenGR00TStateDual.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/`

Changes:
- Added `QwenGR00TStateDual`, a `QwenGR00TState` variant with two independent GR00T DiT action heads.
- The arm head supervises action dimensions `[0:16]`; the camera head supervises action dimensions `[16:18]`; loss is dimension-normalized and inference concatenates both predictions back to 18 dimensions.
- Added `gr00tdual_subtask_action_12_ws`, a cotrain managed run based on `gr00t_subtask_action_12_ws`.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/model/framework/VLM4A/QwenGR00TStateDual.py`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/run_policy_server.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/start_train.sh`
- Parsed `gr00tdual_subtask_action_12_ws/config.yaml` with `OmegaConf`.
- Imported `QwenGR00TStateDual` in the `starVLA` conda environment and confirmed registry registration.
- `git diff --check`

## 2026-07-06 - FAST Tokenizer Offline Loading

Scope:
- `starVLA/model/modules/action_model/fast_ActionHeader.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/README.md`
- `doc/training_config_guidelines.md`

Changes:
- Made the FAST action tokenizer load from local files instead of falling back to a Hugging Face network download when the batch node is offline.
- Added config support for `framework.action_model.fast_tokenizer_name` / `fast_tokenizer_path`.
- Pointed `fast_subtask_action_12_ws` at `./playground/Pretrained_models/fast`.
- Prepared the local FAST tokenizer mirror at `/data/lmz/ckpt/fast` and linked it from `playground/Pretrained_models/fast`.
- Documented the local FAST tokenizer requirement.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/model/modules/action_model/fast_ActionHeader.py`
- `conda run -n starVLA python -c "from omegaconf import OmegaConf; from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model; cfg=OmegaConf.load('examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/config.yaml'); model=get_action_model(cfg); print(type(model.fast_tokenizer).__name__)"`.
- `HF_HUB_OFFLINE=1 conda run -n starVLA python -c "from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer; m=Fast_Action_Tokenizer(); print(type(m.fast_tokenizer).__name__, m.fast_tokenizer.vocab_size)"`.
- `git diff --check`.

## 2026-07-05 - FAST 12-Frame State Local Start Script

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/start_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/README.md`

Changes:
- Added the run-local `start_train.sh` wrapper for `fast_subtask_action_12_ws`.
- Documented the local launcher usage with `GPU_IDS`.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/start_train.sh`
- `git diff --check`

## 2026-07-05 - Planner OFT Training Project

Scope:
- `starVLA/dataloader/planner_oft_datasets.py`
- `starVLA/model/framework/VLM4A/QwenPlannerVLM.py`
- `starVLA/training/train_planner_oft_vlm.py`
- `starVLA/training/train_planner_oft_vla.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/`

Changes:
- Added a separate planner-OFT dataloader module with fixed-stride planner inputs, retrieval-index supervision, and unbounded planner-memory VLA inputs.
- Added a planner-only Qwen VLM framework and two wrapper entrypoints that use the new dataloaders without editing the existing dataloader registry or base training scripts.
- Added the `planner_oft` managed project with planner and VLA configs, train launchers, `yhbatch` submission, policy-server script, and README.

Validation:
- `conda run -n starVLA python -m py_compile starVLA/dataloader/planner_oft_datasets.py starVLA/model/framework/VLM4A/QwenPlannerVLM.py starVLA/training/train_planner_oft_vlm.py starVLA/training/train_planner_oft_vla.py`
- `bash -n` for all `planner_oft` shell scripts.
- Parsed both `planner_oft` YAML configs with `OmegaConf`.
- Built one task-1 VLA memory sample and one task-1 planner VLM batch.
- Checked frame matching examples for retrieval indices.

Notes:
- Runtime planner inference, memory-bank maintenance, and frame-aligned state history remain client/eval-adapter responsibilities.

## 2026-07-05 - Planner OFT Local Start Scripts

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/start_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/start_planner_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/start_vla_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/README.md`

Changes:
- Added run-local local-start wrappers for both planner and VLA stages.
- Kept stage selection in `PLANNER_OFT_STAGE` and delegated environment setup/logging to the shared managed-run launcher.

Validation:
- `bash -n` for the new start scripts.

## 2026-07-05 - FAST 12-Frame State Managed Run

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_12_ws/`

Changes:
- Added `fast_subtask_action_12_ws`, a `QwenFastState` run with action-keyframe history, 12 history frames, subtask think supervision, and state input enabled.
- Configured the run to start from `Qwen3-VL-2B-Instruct-Action` with FAST action tokens already present and `auto_add_action_tokens: false`.
- Added managed-run training, `yhbatch`, policy-server, and README files.

Validation:
- `bash -n` for the added shell scripts.
- Parsed `config.yaml` with `OmegaConf` in the `starVLA` conda environment and checked run id, framework, state flag, history length, no VLM data block, and FAST action-token setting.
- Confirmed `run_train.sh` does not pass framework/datasets/trainer/run/W&B dotlist overrides.
- `git diff --check`.

Notes:
- Set `datasets.vla_data.per_device_batch_size: 6`, matching the existing 12-frame FAST run and keeping margin for state tokens.

## 2026-07-05 - OFT No No Local Start Script

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/start_train.sh`

Changes:
- Added the run-local `start_train.sh` wrapper for `oft_no_no_0_wos`, matching the existing OFT managed-run launcher pattern.
- The wrapper delegates to `../start_managed_train.sh` and does not change training semantics.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/start_train.sh`

Notes:
- Use `GPU_IDS=0,1 ./start_train.sh` from the run directory to bind local startup to a GPU subset.

## 2026-07-05 - GR00T State Action-Keyframe Managed Runs

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/gr00t_no_action_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/gr00t_subtask_action_12_ws/`

Changes:
- Added `gr00t_no_action_12_ws`, a `QwenGR00TState` VLA-only run based on `oft_no_action_12_ws`.
- Added `gr00t_subtask_action_12_ws`, a `QwenGR00TState` cotrain run based on `oft_subtask_action_12_ws`.
- Both runs use action-keyframe history with max 12 history frames, state soft tokens, Qwen3-VL-2B, and a GR00T DiT action head.
- Added run-local training, batch submission, policy server, local start, and README files for both runs.

Validation:
- `bash -n` for all added shell scripts.
- Parsed both new `config.yaml` files with `OmegaConf`.
- Confirmed `run_train.sh` scripts do not pass framework/datasets/trainer dotlist overrides.

## 2026-07-04 - Managed Run W&B Env Precedence

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/start_managed_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/*/submit_yhbatch.sh`

Changes:
- Updated local and batch managed-run launchers to source the repository-root `.env` before defaulting `WANDB_MODE` to `offline`.
- Preserved explicit shell overrides such as `WANDB_MODE=offline` or `WANDB_MODE=disabled` as the highest-priority setting.

Validation:
- `bash -n` for all managed-run `run_train.sh`, `submit_yhbatch.sh`, and the shared local launcher.
- `git diff --check`.

Notes:
- Existing already-started training processes keep their original W&B mode; restart the run to pick up this launcher change.

## 2026-07-04 - OFT Cotrain Local Start Scripts

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/start_managed_train.sh`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_*_*/start_train.sh`

Changes:
- Added a shared local training launcher that infers `NUM_PROCESSES` from `NUM_PROCESSES`, `GPU_IDS`, `CUDA_VISIBLE_DEVICES`, scheduler GPU variables, or `nvidia-smi`.
- Added per-run `start_train.sh` wrappers for the nine OFT cotrain tasks.
- The launcher writes stdout/stderr to each managed run's `logs/` directory and calls the existing `run_train.sh` without changing training semantics.

Validation:
- `bash -n` for the shared launcher and all nine per-run `start_train.sh` wrappers.
- `git diff --check`.

Notes:
- `GPU_IDS=0,1 ./start_train.sh` can be used to bind a run to a specific GPU subset; otherwise all currently visible GPUs are used.

## 2026-07-04 - OFT Cotrain Batch Safety Margin

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_*_*/config.yaml`

Changes:
- Set the OFT cotrain managed-run VLA and VLM per-device batch sizes to the measured maximum OK batch size minus 1.
- Applied the same value to `datasets.vla_data.per_device_batch_size` and `datasets.vlm_data.per_device_batch_size` in each cotrain config.

Validation:
- Parsed all updated OFT cotrain configs with `OmegaConf` in the `starVLA` conda environment.
- `git diff --check`.

Notes:
- `oft_subtask_no_0_ws` previously reached the probe cap at 32 without an OOM, so it was conservatively set to 31 rather than an unmeasured higher value.

## 2026-07-04 - OFT Subtask History Batch Retest

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_12_ws/config.yaml`
- `doc/memory_tests/run_oft_cotrain_2gpu_batch_probe.sh`

Changes:
- Changed the OFT cotrain memory probe default to `REQUIRE_FULL_HISTORY=false`, matching normal training behavior where sparse history can contain fewer than `history.max_frames` frames.
- Added `CASES_OVERRIDE` support so a single managed run can be retested without editing the probe script.
- Retested `oft_subtask_subtask_12_ws` without full-history forcing and set both VLA and VLM per-device batch sizes to 9.

Validation:
- `bash -n doc/memory_tests/run_oft_cotrain_2gpu_batch_probe.sh`.
- Probe report: `doc/memory_tests/oft_cotrain_2gpu_batch_size_20260704T_subtask12_nofull.md`.
- `oft_subtask_subtask_12_ws` passed 5 train steps plus eval at batch size 9; batch size 10 failed with CUDA OOM.

Notes:
- The previous full-history probe was a strict worst-case check and was not representative for `subtask_keyframe` history because the dataset often has fewer than 12 prior subtask keyframes.

## 2026-07-04 - OFT Cotrain Batch Sizes

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_*_*/config.yaml`

Changes:
- Updated OFT cotrain managed-run `datasets.vla_data.per_device_batch_size` and `datasets.vlm_data.per_device_batch_size` from the 2-GPU binary-search memory probe.
- Set VLA and VLM per-device batch sizes to the same tested value for each cotrain run.
- Left `oft_subtask_subtask_12_ws` at batch size 1 because it failed in data sampling, not GPU memory, even at batch size 1.

Validation:
- 2-GPU probe report: `doc/memory_tests/oft_cotrain_2gpu_batch_size_20260704T151930Z.md`.
- `conda run -n starVLA python -c "from omegaconf import OmegaConf; ..."` for all updated OFT cotrain configs.
- `git diff --check`.

Notes:
- `oft_subtask_no_0_ws` reached the configured probe cap of 32, so its true max may be higher.
- `oft_subtask_subtask_12_ws` needs data/history constraint review before batch-size tuning is meaningful.

## 2026-07-04 - OFTState VLM State Cotrain

Scope:
- `starVLA/dataloader/vlm_datasets.py`
- `starVLA/model/framework/VLM4A/QwenOFTState.py`
- `starVLA/training/train_starvla_cotrain.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_instruction_action_12_ws/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_*_ws/config.yaml`
- `doc/training_config_guidelines.md`

Changes:
- Added `datasets.vlm_data.include_state: true` support for LeRobot-backed VLM think batches.
- Preserved per-frame `state_history` through the VLM think dataset and collator.
- Added `QwenOFTState.prepare_vlm_state_conditioned_inputs`, which inserts state soft tokens after the user image/language prompt and before the user turn closes, while masking their labels with `IGNORE_INDEX`.
- Routed cotrain VLM loss through the state-conditioned path when `datasets.vlm_data.include_state` is enabled.
- Enabled VLM state input for the existing `QwenOFTState` cotrain managed runs.

Validation:
- `/data/lmz/miniconda3/envs/starVLA/bin/python -m py_compile starVLA/dataloader/vlm_datasets.py starVLA/model/framework/VLM4A/QwenOFTState.py starVLA/training/train_starvla_cotrain.py`
- Parsed all 11 OFT managed-run `config.yaml` files with `OmegaConf`.
- Built LeRobot-backed VLM think samples for `oft_subtask_no_0_ws`, `oft_subtask_action_6_ws`, `oft_subtask_motion_6_ws`, and `oft_subtask_subtask_6_ws`; verified `state_history` aligns with image count and state-token insertion uses the user prompt boundary.
- `bash -n` for all OFT managed-run `run_train.sh`, `submit_yhbatch.sh`, and `run_policy_server.sh` scripts.
- `git diff --check`.

Notes:
- VLM state conditioning is explicit through `datasets.vlm_data.include_state`; plain `QwenOFT` cotrain runs remain text/image-only on the VLM branch.

## 2026-07-04 - OFT Prompt/History Managed Runs

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_action_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_instruction_action_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_action_12_wos/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_motion_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_motion_6_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_6_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/`
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_action_6_ws/`
- `starVLA/model/framework/VLM4A/QwenOFT.py`
- `starVLA/model/framework/VLM4A/QwenOFTState.py`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added OFT managed runs for no-cotrain/task-cotrain/subtask-cotrain variants, action/motion/subtask history keyframes, 6/12 history lengths, and with/without state variants requested for Astribot.
- Kept all new runs on plain `Qwen3-VL-2B-Instruct`, because `QwenOFT` uses MLP regression over query tokens instead of FAST `<robot_action_*>` token supervision.
- Set conservative `per_device_batch_size: 1` for the history-heavy OFT runs.
- Kept the OFT action branch prompt source fixed to total task instruction through `datasets.vla_data.oft_instruction.source: instruction`.
- Configured `oft_no_action_12_ws` to use `train_starvla.py` without `datasets.vlm_data`, so it does not run the VLM cotrain branch.
- Configured `oft_instruction_*` and `oft_subtask_*` runs to use `train_starvla_cotrain.py` with LeRobot-backed VLM think supervision.
- Set VLM branch supervision through `datasets.vlm_data.think_answer.instruction_source`: `instruction` for `oft_instruction_*`, `subtask_instruction` for `oft_subtask_*`.
- Matched `datasets.vlm_data.CoT_prompt` to `datasets.vla_data.CoT_prompt` in cotrain runs so the VLM and VLA branches receive the same user prompt wording.
- Kept plain `QwenOFT` as the no-state OFT implementation and added `QwenOFTState` for `*_ws` runs.
- `QwenOFTState` projects each frame-aligned state vector through an MLP to one Qwen soft token and inserts those tokens after prompt embeddings and before OFT action-query tokens.
- Reused the managed-run launcher, `yhbatch`, policy-server, and README layout required by the training config docs.

Validation:
- `bash -n` for `run_train.sh`, `submit_yhbatch.sh`, and `run_policy_server.sh` in all nine requested managed runs.
- Parsed all nine `config.yaml` files with `OmegaConf` and verified run id, state flag, framework name, history mode, history length, and OFT instruction source.
- Verified each cotrain run has identical `datasets.vlm_data.CoT_prompt` and `datasets.vla_data.CoT_prompt`.
- `conda run -n starVLA python -m py_compile starVLA/model/framework/VLM4A/QwenOFT.py starVLA/model/framework/VLM4A/QwenOFTState.py`
- Built one `robotwin_astribot_task1` sample for action, motion, and subtask history modes.
- Verified OFT prompt-source selection returns task instruction for the requested OFT action runs.

Notes:
- OFT run names use the second field for cotrain semantics: `no` = no VLM cotrain branch, `instruction` = VLM branch supervised by total task instruction, `subtask` = VLM branch supervised by subtask instruction.

## 2026-07-04 - Managed Run oft_subtask_no_0_ws

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/`
- `starVLA/dataloader/vlm_datasets.py`
- `starVLA/model/framework/share_tools.py`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added an OFT cotrain managed run for RoboTwin Astribot.
- Set `run_id: oft_subtask_no_0_ws`.
- Enabled VLA state input through `datasets.vla_data.include_state: true`.
- Disabled history frames for both VLA and VLM branches.
- Added a LeRobot-backed VLM think-supervision dataloader path for Astribot, avoiding a generated static JSON file.
- Configured the VLM branch to supervise `<think>Frames: ... Now the task is "{subtask_lang}"</think>`.
- Set the run launcher to use `starVLA/training/train_starvla_cotrain.py`.
- Removed the leading space before QwenOFT action-query tokens so Qwen3 tokenizes all 16 query markers as the configured action token.
- Updated checkpoint config loading to prefer `config.full.yaml` when present, because eval needs prompt/data fields that may be omitted from the accessed-only `config.yaml`.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/run_policy_server.sh`
- Parsed `config.yaml` with `OmegaConf`.
- `conda run -n starVLA python -m py_compile starVLA/dataloader/vlm_datasets.py`
- Built one `robotwin_astribot_task1` VLM think batch with the new LeRobot-backed dataloader.
- Ran a one-step `oft_subtask_no_0_ws` cotrain smoke test on one A800 with task1 overrides.
- Loaded the smoke checkpoint through the policy wrapper and ran one eval-style `predict_action`.
- `git diff --check`.

## 2026-07-03 - Managed Run oft_no_no_0_wos

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/`
- `starVLA/model/framework/VLM4A/QwenOFT.py`
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/model/modules/vlm/QWen2_5.py`

Changes:
- Added a pure `QwenOFT` managed run for RoboTwin Astribot.
- Set `run_id: oft_no_no_0_wos`.
- Disabled VLM think supervision, history frames, and state input.
- Set the base VLM to plain `Qwen3-VL-2B-Instruct` instead of the FAST `*-Action` checkpoint.
- Changed QwenOFT prompt assembly so the configured `CoT_prompt` remains the natural-language prompt and OFT action-query tokens are appended after the full prompt without `<action>` tags.
- Added optional `prompt_suffixes` support to Qwen3/Qwen2.5 VLM input builders.
- Updated the OFT prompt wording to describe image order as earliest-to-latest with the last image as the current view.
- Reused the RoboTwin Astribot LeRobot data mix and no-history managed-run script layout.

Validation:
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/oft_no_no_0_wos/run_policy_server.sh`
- Parsed `config.yaml` with `OmegaConf`.
- `git diff --check`.

## 2026-07-02 - fast_subtask_action_6_wos_test1 Fresh Start

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/README.md`

Changes:
- Set `trainer.is_resume: false` for `fast_subtask_action_6_wos_test1`.
- Kept `trainer.pretrained_checkpoint` unset, so this run starts from the configured base VLM instead of a managed-run checkpoint.
- Updated the run README to document fresh-start behavior.

Validation:
- Parsed the updated `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_policy_server.sh`
- `git diff --check`.

## 2026-07-02 - QwenFast Decode Failure Long Preview

Scope:
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/config.yaml`

Changes:
- Increased QwenFast decode-failure `text_preview` and `actual_action_text_preview` from 500 chars to configurable 4096 chars for the test run.
- Added tail preview, character counts, and truncation flags to decode-failure details.
- For missing `</action>` failures, now records the partial text after `<action>` in `actual_action_text_preview`.
- Added `has_open_action_tag` and `has_close_action_tag` flags for incomplete action-block failures.

Validation:
- `python -m py_compile starVLA/model/framework/VLM4A/QwenFast.py`
- Confirmed `fast_decode_failure_preview_chars` and `fast_decode_failure_tail_chars` are present in the test run config with `rg`.
- `git diff --check`.

## 2026-07-02 - QwenFast Eval Decode Failure Detail

Scope:
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `starVLA/training/train_starvla.py`

Changes:
- Passed eval target FAST token ids into `predict_action`.
- Expanded FAST decode failure details with actual token count, actual decoded coefficient count, actual action text preview, target token count, target decoded coefficient count, and target token preview.
- Added the same target-vs-actual detail for missing, empty, or out-of-range `<action>` blocks.
- Added `text_preview` to every decode failure detail and log the full detail on every eval decode failure instead of only the first failure.

Validation:
- `python -m py_compile starVLA/model/framework/VLM4A/QwenFast.py starVLA/training/train_starvla.py`
- `git diff --check`.

## 2026-07-02 - QwenFast Action-Block Token Extraction

Scope:
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `starVLA/training/train_starvla.py`

Changes:
- Changed QwenFast eval/inference action token extraction to parse only the generated `<action>...</action>` block.
- Extracted FAST ids from `<robot_action_N>` strings inside that block instead of filtering all action-token ids from the full generated sequence.
- Treated missing, empty, or out-of-range action blocks as action decode failures that eval can skip without crashing training.

Validation:
- `python -m py_compile starVLA/model/framework/VLM4A/QwenFast.py starVLA/training/train_starvla.py`
- Ran a lightweight extractor check confirming tokens outside `<action>...</action>` are ignored and missing action blocks raise the expected decode failure.
- `git diff --check`.

## 2026-07-02 - fast_subtask_action_6_wos_test1 Resume Mode

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/config.yaml`
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/README.md`

Changes:
- Set `trainer.is_resume: true` for `fast_subtask_action_6_wos_test1`.
- Kept `trainer.pretrained_checkpoint` unset, so resume uses only the run-local latest checkpoint.
- Documented that missing run-local checkpoints fall back to the configured base VLM.

Validation:
- Parsed the updated `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_policy_server.sh`
- `git diff --check`.

## 2026-07-02 - QwenFast Eval Generation Headroom

Scope:
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `starVLA/training/train_starvla.py`

Changes:
- Changed QwenFast action eval generation from `max_length=2048` to `max_new_tokens`, so long multi-image prompts do not consume the action-token generation budget.
- Kept deterministic eval generation with `do_sample: false`.
- Logged the first FAST decode failure detail, including whether generation produced no action tokens or an invalid decoded coefficient count.

Validation:
- `python -m py_compile starVLA/model/framework/VLM4A/QwenFast.py starVLA/training/train_starvla.py`
- `git diff --check`.

## 2026-07-02 - Managed Run fast_subtask_action_6_wos_test1

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/`
- `examples/RoboTwin_Astribot/train_files/data_registry/data_config.py`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added a debug managed run derived from `fast_subtask_action_6_wos`.
- Set `run_id: fast_subtask_action_6_wos_test1`.
- Set `datasets.vla_data.data_mix: robotwin_astribot_task1`.
- Set `robotwin_astribot_task1` to only `beat_block_hammer_rotate_view`.
- Kept `QwenFast`, no-state input, and 6 history frames.
- Removed `trainer.pretrained_checkpoint`, so this run does not initialize from `fast_subtask_action_12_wos`.

Validation:
- Parsed the new `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_action_6_wos_test1/run_policy_server.sh`
- `git diff --check`.

## 2026-07-01 - QwenFastState State Encoder BF16 Fix

Scope:
- `starVLA/model/framework/VLM4A/QwenFastState.py`

Changes:
- Cast `StateHistoryEncoder` inputs to the encoder parameter dtype/device before LayerNorm and MLP projection.
- Run the state encoder projection with autocast disabled to avoid mixed `float32`/`bfloat16` LayerNorm failures during eval and inference.
- Cast frame-position state embeddings to the projected state-token dtype before addition.

Validation:
- `python -m py_compile starVLA/model/framework/VLM4A/QwenFastState.py`
- Instantiated `StateHistoryEncoder` in both `float32` and `bfloat16`; verified a float32 state input produces matching output dtype without LayerNorm dtype errors.

## 2026-07-01 - Managed Run fast_subtask_no_0_ws

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added a managed run for `fast_subtask_no_0_ws`.
- Set `framework.name: QwenFastState` and `datasets.vla_data.include_state: true`.
- Added `framework.state_model` with 18-D state inputs and one state token per current-frame sample.
- Disabled history with `history.enabled: false` and `history.max_frames: 0`; kept subtask `<think>` supervision and the offline `Qwen3-VL-2B-Instruct-Action` base VLM.
- Added `run_train.sh`, `submit_yhbatch.sh`, `run_policy_server.sh`, and run-local README using the managed run conventions.

Validation:
- Parsed `config.yaml` with `OmegaConf` using `/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/starVLA/bin/python`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_ws/run_policy_server.sh`
- Confirmed `run_train.sh` does not pass training semantic dotlist overrides.
- `git diff --check`.
- Checked the new untracked run files for trailing whitespace.

## 2026-06-30 - Managed Run fast_subtask_no_0_wos

Scope:
- `examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/`
- `examples/RoboTwin_Astribot/README.md`

Changes:
- Added a managed run for `fast_subtask_no_0_wos`.
- Set `framework.name: QwenFast` and `datasets.vla_data.include_state: false`.
- Kept subtask `<think>` supervision and the offline `Qwen3-VL-2B-Instruct-Action` base VLM.
- Disabled history with `history.enabled: false` and `history.max_frames: 0`; `no` is only the run-name history-keyframe field.
- Added `run_train.sh`, `submit_yhbatch.sh`, `run_policy_server.sh`, and run-local README using the managed run conventions.

Validation:
- Parsed `config.yaml` with `OmegaConf`.
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/run_train.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/submit_yhbatch.sh`
- `bash -n examples/RoboTwin_Astribot/train_files/managed_runs/fast_subtask_no_0_wos/run_policy_server.sh`
- `git diff --check`.

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

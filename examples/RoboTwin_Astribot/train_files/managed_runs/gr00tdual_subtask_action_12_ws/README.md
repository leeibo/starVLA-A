# gr00tdual_subtask_action_12_ws

This managed run trains `QwenGR00TStateDual` on Astribot LeRobot data with:

- base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- action heads: two GR00T DiT flow-matching heads
- arm action head: first 16 action dimensions
- camera action head: final 2 action dimensions
- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by subtask instruction
- history: `action_keyframe`, max 12 history frames
- state input: enabled (`include_state: true`), one MLP-projected soft token per input image

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/run_train.sh
```

Submit:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_ws/run_policy_server.sh
```

The run starts from the plain Qwen3-VL checkpoint and does not use FAST action
tokens. `QwenGR00TStateDual` conditions state through Qwen soft tokens and
disables the GR00T action heads' internal state branch. The VLM branch is
supervised by `datasets.vlm_data.think_answer.instruction_source:
subtask_instruction`.

Outputs are written to `results/Checkpoints/gr00tdual_subtask_action_12_ws`.

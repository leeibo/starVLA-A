# gr00tdual_subtask_action_12_wos

This managed run trains `QwenGR00TDual` on Astribot LeRobot data with:

- base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- action heads: two GR00T DiT flow-matching heads
- arm action head: first 16 action dimensions
- camera action head: final 2 action dimensions
- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by subtask instruction
- history: `action_keyframe`, max 12 history frames
- state input: disabled (`include_state: false`)

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_wos/run_train.sh
```

Submit:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_wos/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00tdual_subtask_action_12_wos/run_policy_server.sh
```

The run starts from the plain Qwen3-VL checkpoint and does not use FAST action
tokens. `QwenGR00TDual` does not consume proprioceptive state; it uses the Qwen
hidden states to condition two independent GR00T action heads and concatenates
their arm/camera predictions back to 18 action dimensions.

Outputs are written to `results/Checkpoints/gr00tdual_subtask_action_12_wos`.

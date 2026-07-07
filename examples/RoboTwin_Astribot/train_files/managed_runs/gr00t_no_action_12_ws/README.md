# gr00t_no_action_12_ws

This managed run trains `QwenGR00TState` on Astribot LeRobot data with:

- base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- action head: GR00T DiT flow-matching action generation
- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: disabled
- history: `action_keyframe`, max 12 history frames
- state input: enabled (`include_state: true`), one MLP-projected soft token per input image

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00t_no_action_12_ws/run_train.sh
```

Submit:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/gr00t_no_action_12_ws/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/gr00t_no_action_12_ws/run_policy_server.sh
```

The run starts from the plain Qwen3-VL checkpoint and does not use FAST action
tokens. `QwenGR00TState` conditions state through Qwen soft tokens and disables
the GR00T action head's internal state branch.

Outputs are written to `results/Checkpoints/gr00t_no_action_12_ws`.

# oft_subtask_no_0_ws

This managed run trains `QwenOFTState` on Astribot LeRobot data with:

- base VLM: `./playground/Pretrained_models/Qwen3-VL-2B-Instruct`
- action head: OFT MLP regression over action query tokens
- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by subtask instruction
- history: disabled (`history.enabled: false`, `history.max_frames: 0`)
- state input: enabled (`include_state: true`), one MLP-projected soft token per input image

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/run_train.sh
```

Submit:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_no_0_ws/run_policy_server.sh
```

The run starts from the plain Qwen3-VL checkpoint and does not use FAST action
tokens. `oft_instruction.source: instruction` keeps the OFT action prompt on
the task instruction; `datasets.vlm_data.think_answer.instruction_source:
subtask_instruction` controls the VLM cotrain supervision.

Outputs are written to `results/Checkpoints/oft_subtask_no_0_ws`.

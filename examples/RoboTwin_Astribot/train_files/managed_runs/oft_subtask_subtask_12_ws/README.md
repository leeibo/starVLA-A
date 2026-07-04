# oft_subtask_subtask_12_ws

This managed run trains `QwenOFTState` on Astribot LeRobot data with:

- VLA action prompt: task instruction (`lang`)
- VLM cotrain branch: enabled, with think output supervised by subtask instruction
- history: `subtask_keyframe`, max 12 history frames
- state input: enabled (`include_state: true`), one MLP-projected soft token per input image

Launch:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_12_ws/run_train.sh
```

Submit:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_12_ws/submit_yhbatch.sh
```

Inference server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_subtask_12_ws/run_policy_server.sh
```

Outputs are written to `results/Checkpoints/oft_subtask_subtask_12_ws`.

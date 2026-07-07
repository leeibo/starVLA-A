# planner_oft

Two-stage planner-driven OFT setup for Astribot LeRobot data.

- planner stage: `QwenPlannerVLM`, fixed stride 16, max 12 history frames plus current, supervised to emit `<subtask>` and `<retrieval>`.
- VLA stage: `QwenOFTState`, subtask prompt, planner-memory frames from the full unbounded stride history plus current.
- dataset code: `starVLA/dataloader/planner_oft_datasets.py`, separate from the existing dataloaders.

Planner training:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/run_planner_train.sh
```

Local planner training wrapper:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/start_planner_train.sh
```

VLA training:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/run_vla_train.sh
```

Local VLA training wrapper:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/start_vla_train.sh
```

Queue planner:

```bash
PLANNER_OFT_STAGE=planner yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/submit_yhbatch.sh
```

Queue VLA:

```bash
yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/submit_yhbatch.sh
```

VLA policy server:

```bash
bash examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft/run_policy_server.sh
```

Outputs:

- planner checkpoint: `results/Checkpoints/planner_oft_planner`
- VLA checkpoint: `results/Checkpoints/planner_oft`

The VLA server only serves the trained OFT policy. Runtime planner calls,
memory-bank maintenance, and frame-aligned `state_history` construction should
stay in the client or eval adapter.

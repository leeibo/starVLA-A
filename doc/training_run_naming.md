# Training Run Naming

训练脚本和配置按以下规则命名：

```text
{architecture}_{think_level}_{history_keyframe}_{memory_length}_{state_flag}
```

字段含义：

- `architecture`: 框架类别。当前 `fast` 表示 Qwen FAST-token 训练族；具体是否启用 state 由 `state_flag` 决定。
- `think_level`: `<think>` 中监督使用的指令级别。`total` 表示总任务 instruction，`subtask` 表示 subtask instruction。
- `history_keyframe`: 历史帧来源。`action` 表示 `action_keyframe`，`motion` 表示 `motion_keyframe`，`subtask` 表示 `subtask_keyframe`，`no` 表示不使用历史帧。
- `memory_length`: 历史帧数量上限。数字表示最多使用对应数量的历史帧，`ultra` 表示不限制历史帧数量。
- `state_flag`: 是否使用本体状态。`wos` 表示 without state，对应 `QwenFast`；`ws` 表示 with state，对应 `QwenFastState`。

示例：

```text
fast_subtask_action_ultra_wos
```

这个名字表示：

- `QwenFast`
- `<think>` 监督使用 subtask instruction
- 历史帧来自 action chunk keyframe
- 历史长度不限制
- 不使用 state

管理约定：

- 新增同类训练时，在 `examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/` 下创建独立子目录。
- 子目录至少包含 `config.yaml`、`run_train.sh`、`submit_yhbatch.sh`、`run_policy_server.sh` 和 `README.md`。
- `config.yaml` 的 `run_id` 应与 `<run_name>` 保持一致。
- 输出目录由 `config.yaml` 中的 `run_root_dir` 和 `run_id` 决定。
- 训练语义写在 `config.yaml`，`run_train.sh` 只负责训练启动，`submit_yhbatch.sh` 只负责排队提交、运行环境和本地 stdout/stderr 日志，`run_policy_server.sh` 只负责启动该 run 的推理服务；具体配置写法见 `doc/training_config_guidelines.md`。
- 涉及训练脚本命名、目录组织或新增 managed run 的需求，先查阅本文档。

## 新建 Managed Run 清单

每次创建新的训练项目时，按以下顺序检查：

- 创建目录：`examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/`。
- 添加 `config.yaml`，并确认 `run_id: <run_name>`。
- 添加 `run_train.sh`，只传 `--config_yaml`，不写训练语义覆盖。
- 添加 `submit_yhbatch.sh`，用于 `yhbatch` 队列提交。
- 添加 `run_policy_server.sh`，用于启动该 run 的 policy server。
- 添加 `README.md`，写明训练目标、启动命令、提交命令、推理 server 命令、输出目录和 checkpoint 来源。
- 运行 `bash -n run_train.sh`、`bash -n submit_yhbatch.sh` 和 `bash -n run_policy_server.sh`。
- 用 `OmegaConf` 解析 `config.yaml`。
- 更新 `doc/CHANGELOG.md`。

`submit_yhbatch.sh` 的最低要求：

- 默认使用固定 repo root，并允许 `STARGVLA_REPO_ROOT` 覆盖。不要依赖 `BASH_SOURCE[0]` 定位 run 目录，因为 batch 系统可能把脚本复制到 `/tmp/slurmd/...` 后执行。
- 默认绑定本地 `starVLA` conda 环境里的 `accelerate`，例如 `/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/starVLA/bin/accelerate`。
- 自动推断 `NUM_PROCESSES`：优先 scheduler GPU 变量，其次 `CUDA_VISIBLE_DEVICES`，最后 `nvidia-smi -L`。
- 默认 `WANDB_MODE=offline`，并允许提交命令显式设置 `WANDB_MODE=disabled/offline/online`。
- 将 stdout/stderr 复制保存到当前 run 目录下的 `logs/`。
- 最后调用同目录的 `run_train.sh`。

`run_policy_server.sh` 的最低要求：

- 默认使用固定 repo root，并允许 `STARGVLA_REPO_ROOT` 覆盖。
- 默认绑定本地 `starVLA` conda 环境里的 `python`，例如 `/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/starVLA/bin/python`。
- 默认从 `results/Checkpoints/<run_name>` 自动选择最新 checkpoint；允许 `POLICY_CKPT_PATH` 显式指定。
- 默认端口为 `7980`，允许 `POLICY_PORT` 覆盖。
- 只启动 `deployment/model_server/server_policy.py`，不承担环境端 history/state 维护。

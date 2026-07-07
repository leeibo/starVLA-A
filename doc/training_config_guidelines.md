# Training Config Guidelines

本文档记录 RoboTwin Astribot 训练配置的写法约定。

## 核心原则

- `config.yaml` 是训练语义的唯一来源。模型、数据、prompt、history、optimizer、trainer schedule、`run_id`、`run_root_dir`、W&B 配置都写在 config 中。
- `run_train.sh` 只负责启动环境。它可以设置 conda/accelerate fallback、`NUM_PROCESSES`、NCCL/runtime 环境变量和 `--config_yaml`，不要默认传入 `--framework.*`、`--datasets.*`、`--trainer.*`、`--run_id`、`--wandb_*` 等 dotlist 覆盖。
- 配置保持最小可读。不要从 cotrain 模板继承无关字段；只保留当前 entrypoint、framework、dataloader、trainer 实际需要的字段。
- `train_starvla.py` 的 FAST VLA 训练不需要 `datasets.vlm_data`。只有使用 VLM-only 或 cotrain entrypoint 时才添加 VLM 数据块。
- YAML 类型要明确：无限历史用 `null`，布尔值用 `true/false`，数字不要写成字符串。

## FAST VLA 配置要点

- `framework.name: QwenFast` 表示不使用 state；`datasets.vla_data.include_state: false`。
- `framework.name: QwenFastState` 表示使用 state；`datasets.vla_data.include_state: true`，并添加 `framework.state_model`。
- FAST 动作监督需要 `<robot_action_*>` token。正式训练配置应指向离线生成的 `*-Action` checkpoint，例如 `./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`，并设置 `framework.action_model.auto_add_action_tokens: false`。
- FAST tokenizer 也必须使用本地目录，避免 batch 节点联网下载 `physical-intelligence/fast`；推荐在 `framework.action_model.fast_tokenizer_name` 中写 `./playground/Pretrained_models/fast`，或通过 `STARGVLA_FAST_TOKENIZER` 指向等价本地目录。
- plain Qwen3/Qwen2.5 checkpoint 只用于显式开发实验；此时才手动设置 `framework.action_model.auto_add_action_tokens: true`，让启动过程临时添加并 resize embedding。
- `datasets.vla_data.dataset_py` 使用 `lerobot_datasets`。
- `datasets.vla_data.CoT_prompt` 定义 user prompt。
- `datasets.vla_data.fast_answer` 定义监督回答模板和 `<think>` 使用的 instruction 来源。
- `datasets.vla_data.history.mode` 可用 `action_keyframe`、`motion_keyframe`、`subtask_keyframe`。
- `history.max_frames: null` 表示不限制历史帧；数字表示最多使用对应数量的历史帧，不含当前帧。
- 0-history 训练使用 `history.enabled: false` 和 `history.max_frames: 0`；不要只写 `history.max_frames: 0` 同时保持 history enabled，因为 action-keyframe 逻辑仍可能采到一个历史帧。

## OFT VLA / Cotrain 配置要点

- `framework.name: QwenOFT` 使用 plain Qwen-VL checkpoint 和 MLP action head，不使用 FAST `<robot_action_*>` token 监督，也不接收 state 输入。
- `framework.name: QwenOFTState` 是 OFT 的 state 版本；`datasets.vla_data.include_state: true`，并添加 `framework.state_model`。state 不写进 prompt 文本，而是经过 MLP 投到 Qwen hidden size，作为每张输入图片对应的一个 soft token 插入到 prompt embedding 后、OFT action query token 前。
- OFT action 分支的 prompt 统一使用总任务 instruction 构造，配置为 `datasets.vla_data.oft_instruction.source: instruction`。
- `oft_no_*` run 使用 `starVLA/training/train_starvla.py`，不配置 `datasets.vlm_data`，表示不启用 VLM branch co-training。
- `oft_instruction_*` 和 `oft_subtask_*` run 使用 `starVLA/training/train_starvla_cotrain.py`，需要同时配置 `datasets.vla_data` 和 `datasets.vlm_data`。
- Cotrain run 中 `datasets.vlm_data.CoT_prompt` 必须与 `datasets.vla_data.CoT_prompt` 保持一致；两边只通过监督答案来源区分 `instruction/subtask`。
- VLM branch 的监督来源由 `datasets.vlm_data.think_answer.instruction_source` 控制：`instruction` 表示总任务 instruction，`subtask_instruction` 表示 subtask instruction。
- `QwenOFTState` cotrain run 可以设置 `datasets.vlm_data.include_state: true`，让 VLM branch 也接收与输入图片一一对应的 state soft token。VLM state token 插在 user 图像/语言 prompt embedding 之后、user turn 结束 token 之前，label 为 `IGNORE_INDEX`；这样 VLM 与 VLA action branch 的输入条件只差 OFT action query token。

## run_train.sh 检查项

- 默认只传 `--config_yaml` 给 `train_starvla.py`。
- 不要在脚本里默认覆盖 batch size、base VLM、data mix、训练步数、保存间隔、W&B、`run_id` 或 `run_root_dir`。
- 需要临时改训练语义时，优先改 run 目录下的 `config.yaml`。
- 如果用户明确要求临时 CLI 覆盖，必须在 README 或 changelog 中记录原因。

## yhbatch 提交脚本

- 每个正式 managed run 必须提供 `submit_yhbatch.sh`，用于排队提交、显式绑定运行环境和保存 stdout/stderr。新增训练项目时应与 `config.yaml`、`run_train.sh`、`README.md` 一起创建。
- `submit_yhbatch.sh` 不写训练语义，不传 `--framework.*`、`--datasets.*`、`--trainer.*`、`--run_id` 或 `--wandb_*` 覆盖；训练语义仍只来自 `config.yaml`。
- `submit_yhbatch.sh` 可以设置 `ACCELERATE_BIN`、`NUM_PROCESSES`、`WANDB_MODE`、`PYTHONUNBUFFERED`、日志目录和资源申请注释。
- `NUM_PROCESSES` 必须等于本次 job 实际可见 GPU 数。提交脚本应优先从 scheduler 环境变量、`CUDA_VISIBLE_DEVICES` 或 `nvidia-smi -L` 自动推断；只有 scheduler 暴露的 GPU 数错误时才手动覆盖。
- batch 系统可能把脚本复制到 `/tmp/slurmd/...` 后再执行，因此 `submit_yhbatch.sh` 不要依赖 `BASH_SOURCE[0]` 来定位 run 目录。使用固定 repo root，或提供 `STARGVLA_REPO_ROOT` 作为显式覆盖。
- `run_train.sh` 应尊重提交环境里显式传入的 `WANDB_MODE`。即使根目录 `.env` 里写了 `WANDB_MODE=online`，`WANDB_MODE=offline yhbatch ...` 或 `WANDB_MODE=disabled yhbatch ...` 也应该优先。

## 推理脚本

- 每个正式 managed run 必须提供 `run_policy_server.sh`，用于启动该 run 对应的 policy server。
- 不需要为 policy server 生成 `submit_policy_server_yhbatch.sh`；如果确实需要排队启动 server，后续按具体需求单独加。
- `run_policy_server.sh` 不写训练语义，不修改 `config.yaml`，只解析 checkpoint 并调用 `deployment/model_server/server_policy.py`。
- 默认 checkpoint 选择顺序：`POLICY_CKPT_PATH` 显式指定；否则 `results/Checkpoints/<run_id>/checkpoints/steps_*` 中最新的模型；否则 `final_model`。
- server 只做模型推理。多图 history 和 `state_history` 的维护由客户端或 eval adapter 负责。

## W&B 配置

- `config.yaml` 中只写 `wandb_project` 和 `wandb_entity`。`wandb_entity` 是 W&B username 或 team slug，不是 API key。
- `run_train.sh` 可以默认 `WANDB_MODE=online`，但不要把明文 `WANDB_API_KEY` 写进脚本或 md 文档。
- 需要免登录启动时，在不纳入版本管理的 `.env` 中写：

```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_MODE=online
```

- managed run 脚本统一加载仓库根目录 `.env`。`.env` 已被 `.gitignore` 忽略。
- 新建 managed run 时复制这个根目录 `.env` 加载逻辑，而不是复制明文 key，也不需要为每个任务单独放 `.env`。

## 离线 Action Checkpoint

- 使用 `starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py` 将 plain Qwen-VL checkpoint 转成 `*-Action` checkpoint。
- RoboTwin Astribot 当前 2B FAST checkpoint:
  `playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action` -> `/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/ckpt/Qwen3-VL-2B-Instruct-Action`。
- Qwen3 FAST token id 范围应为 `[151669, 153716]`，共 2048 个 token。
- 生成后用 tokenizer 校验 `<robot_action_0>` 到 `<robot_action_2047>` 是否全部存在，再更新训练 YAML。

## 验证

常用检查：

```bash
bash -n examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/run_train.sh
bash -n examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/run_policy_server.sh
conda run -n starVLA python -c "from omegaconf import OmegaConf; OmegaConf.load('examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/config.yaml')"
rg -n -- "--(framework|datasets|trainer|run_root_dir|run_id|wandb_)" examples/RoboTwin_Astribot/train_files/managed_runs/<run_name>/run_train.sh
git diff --check
```

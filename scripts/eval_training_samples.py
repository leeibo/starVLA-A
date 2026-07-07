#!/usr/bin/env python3
"""Evaluate latest checkpoints on samples from their own training data."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


STEP_RE = re.compile(r"steps_(\d+)_")


def checkpoint_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    candidates: list[Path] = []
    if ckpt_dir.exists():
        candidates.extend(ckpt_dir.glob("steps_*_pytorch_model.pt"))
        candidates.extend(ckpt_dir.glob("steps_*_model.safetensors"))
    if candidates:
        return max(candidates, key=checkpoint_step)

    for candidate in (run_dir / "final_model" / "pytorch_model.pt", run_dir / "final_model" / "model.safetensors"):
        if candidate.exists():
            return candidate
    return None


def config_path_for_checkpoint(ckpt_path: Path) -> Path:
    run_dir = ckpt_path.parents[1]
    full_cfg = run_dir / "config.full.yaml"
    return full_cfg if full_cfg.exists() else run_dir / "config.yaml"


def load_config(ckpt_path: Path):
    cfg = OmegaConf.load(config_path_for_checkpoint(ckpt_path))
    apply_config_compat(cfg)
    if not hasattr(cfg, "output_dir"):
        cfg.output_dir = str(ckpt_path.parents[1])
    if hasattr(cfg, "trainer"):
        cfg.trainer.pretrained_checkpoint = None
    return cfg


def load_model(ckpt_path: Path, cfg, device: str):
    model = build_framework(cfg)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path))
    else:
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def build_vla_dataloader(cfg, batch_size: int) -> DataLoader:
    data_cfg = cfg.datasets.vla_data
    data_cfg.per_device_batch_size = int(batch_size)
    data_cfg.num_workers = 0
    data_cfg.pin_memory = False

    dataset_py = str(data_cfg.dataset_py)
    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        dataset = get_vla_dataset(
            data_cfg=data_cfg,
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
        )
    elif dataset_py == "planner_oft_datasets":
        from starVLA.dataloader.planner_oft_datasets import collate_fn, get_planner_oft_dataset

        dataset = get_planner_oft_dataset(
            data_cfg=data_cfg,
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
        )
    else:
        raise ValueError(f"Unsupported VLA dataset_py={dataset_py!r}")

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0, pin_memory=False)


def build_vlm_dataloader(cfg, batch_size: int) -> DataLoader:
    data_cfg = cfg.datasets.vlm_data
    data_cfg.per_device_batch_size = int(batch_size)
    data_cfg.num_workers = 0
    data_cfg.pin_memory = False

    dataset_py = str(data_cfg.dataset_py)
    if dataset_py == "planner_oft_datasets":
        from starVLA.dataloader.planner_oft_datasets import make_planner_oft_vlm_dataloader

        return make_planner_oft_vlm_dataloader(cfg)
    if dataset_py == "vlm_datasets":
        from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

        return make_vlm_dataloader(cfg)["train_dataloader"]
    raise ValueError(f"Unsupported VLM dataset_py={dataset_py!r}")


def move_to_device(value: Any, device: str):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def action_targets(batch: list[dict], horizon: int) -> np.ndarray:
    actions = np.asarray([example["action"] for example in batch], dtype=np.float32)
    return actions[:, -int(horizon) :, :]


def target_fast_tokens(model, batch: list[dict]):
    action_model = getattr(model, "action_model", None)
    if action_model is None or not hasattr(action_model, "encoder_action2fastoken"):
        return None
    return action_model.encoder_action2fastoken([example["action"] for example in batch])


def eval_vla_run(model, cfg, device: str, *, batch_size: int, max_samples: int, do_predict: bool, max_new_tokens: int):
    loader = build_vla_dataloader(cfg, batch_size=batch_size)
    total = 0
    losses: list[float] = []
    l1_values: list[float] = []
    mse_values: list[float] = []
    decode_failures = 0
    predict_errors: list[str] = []
    preview: dict[str, Any] | None = None

    for batch in loader:
        if total >= max_samples:
            break
        batch = batch[: max_samples - total]
        with torch.inference_mode():
            out = model.forward(batch)
        loss = out.get("action_loss", None)
        if loss is not None:
            losses.append(float(loss.detach().float().cpu().item()))

        if do_predict:
            try:
                pred_kwargs = {"max_new_tokens": max_new_tokens}
                fast_tokens = target_fast_tokens(model, batch)
                if fast_tokens is not None:
                    pred_kwargs["target_fast_tokens"] = fast_tokens
                with torch.inference_mode():
                    pred_out = model.predict_action(examples=batch, **pred_kwargs)
                pred = np.asarray(pred_out["normalized_actions"], dtype=np.float32)
                target = action_targets(batch, pred.shape[1])
                diff = pred - target
                l1_values.append(float(np.mean(np.abs(diff))))
                mse_values.append(float(np.mean(np.square(diff))))
                if preview is None:
                    preview = {
                        "target_first_action": target[0, 0, : min(6, target.shape[-1])].round(4).tolist(),
                        "pred_first_action": pred[0, 0, : min(6, pred.shape[-1])].round(4).tolist(),
                    }
            except Exception as exc:  # keep later checkpoints running
                decode_failures += len(batch)
                predict_errors.append(str(exc)[:2000])

        total += len(batch)

    result: dict[str, Any] = {
        "mode": "vla",
        "samples": total,
        "action_loss_mean": float(np.mean(losses)) if losses else None,
        "predict_l1_mean": float(np.mean(l1_values)) if l1_values else None,
        "predict_mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "predict_failures": decode_failures,
    }
    if predict_errors:
        result["predict_error"] = predict_errors[0]
    if preview is not None:
        result["preview"] = preview
    return result


def eval_vlm_run(model, cfg, device: str, *, batch_size: int, max_samples: int):
    loader = build_vlm_dataloader(cfg, batch_size=batch_size)
    total = 0
    losses: list[float] = []

    for batch in loader:
        if total >= max_samples:
            break
        batch_size_actual = int(batch["input_ids"].shape[0]) if isinstance(batch, dict) and "input_ids" in batch else 1
        if total + batch_size_actual > max_samples:
            # VLM collators produce tensors; keep the first partial batch.
            keep = max_samples - total
            batch = {
                key: value[:keep] if torch.is_tensor(value) and value.shape[0] == batch_size_actual else value
                for key, value in batch.items()
            }
            batch_size_actual = keep
        batch = move_to_device(batch, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.qwen_vl_interface(**batch, return_dict=True)
        losses.append(float(out.loss.detach().float().cpu().item()))
        total += batch_size_actual

    return {
        "mode": "vlm",
        "samples": total,
        "vlm_loss_mean": float(np.mean(losses)) if losses else None,
    }


def eval_one(run_name: str, ckpt_path: Path, args) -> dict[str, Any]:
    start = time.time()
    cfg = load_config(ckpt_path)
    framework = str(cfg.framework.name)
    device = args.device
    result: dict[str, Any] = {
        "run": run_name,
        "framework": framework,
        "checkpoint": str(ckpt_path),
        "step": checkpoint_step(ckpt_path),
        "status": "ok",
    }

    model = None
    try:
        model = load_model(ckpt_path, cfg, device=device)
        if hasattr(cfg.datasets, "vla_data") and hasattr(model, "predict_action"):
            metrics = eval_vla_run(
                model,
                cfg,
                device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                do_predict=not args.no_predict,
                max_new_tokens=args.max_new_tokens,
            )
        elif hasattr(cfg.datasets, "vlm_data"):
            metrics = eval_vlm_run(
                model,
                cfg,
                device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
        else:
            raise ValueError("No supported datasets.vla_data or datasets.vlm_data section found")
        result.update(metrics)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=20)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["elapsed_sec"] = round(time.time() - start, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/Checkpoints")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(args.checkpoint_root)
    runs = args.runs or [path.name for path in sorted(root.iterdir()) if path.is_dir()]

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for run_name in runs:
        run_dir = root / run_name
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            continue
        print(json.dumps({"event": "start", "run": run_name, "checkpoint": str(ckpt)}, ensure_ascii=False), flush=True)
        result = eval_one(run_name, ckpt, args)
        all_results.append(result)
        line = json.dumps(result, ensure_ascii=False)
        print(line, flush=True)
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    if output_path is not None:
        summary_path = output_path.with_suffix(".summary.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    return 0 if all(item.get("status") == "ok" for item in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

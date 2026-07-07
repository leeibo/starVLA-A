#!/usr/bin/env python3
"""Export VLM-branch sample inputs, model outputs, and images for inspection."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.vlm_datasets import (
    DataCollatorForSupervisedDataset,
    LeRobotThinkDataset,
    _qwen_user_content_end_index,
    update_processor_pixels,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


STEP_RE = re.compile(r"steps_(\d+)_")


def checkpoint_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    ckpt_dir = run_dir / "checkpoints"
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
    return model.to(device).eval()


def has_vlm_branch(cfg) -> bool:
    return hasattr(cfg, "datasets") and hasattr(cfg.datasets, "vlm_data")


def scan_vlm_runs(root: Path) -> list[str]:
    runs: list[str] = []
    for cfg_path in sorted(root.glob("*/config.full.yaml")):
        cfg = OmegaConf.load(cfg_path)
        if has_vlm_branch(cfg) and latest_checkpoint(cfg_path.parent) is not None:
            runs.append(cfg_path.parent.name)
    return runs


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def jsonable(value: Any):
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def move_to_device(value: Any, device: str):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def tensor_shape(value: Any):
    if value is None:
        return None
    if torch.is_tensor(value):
        return list(value.shape)
    arr = np.asarray(value)
    return list(arr.shape)


def image_list(sample: dict) -> list:
    images = sample["image"]
    return images if isinstance(images, list) else [images]


def save_images(images: list, sample_dir: Path) -> list[str]:
    image_dir = sample_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, image in enumerate(images):
        path = image_dir / f"frame_{idx:03d}.png"
        image.save(path)
        paths.append(str(path.relative_to(sample_dir)))
    return paths


def build_data_args(cfg):
    data_args = SimpleNamespace(**OmegaConf.to_container(cfg.datasets.vlm_data, resolve=True))
    data_args.data_flatten = getattr(data_args, "data_flatten", False)
    data_args.data_packing = getattr(data_args, "data_packing", False)
    return data_args


def build_think_dataset(model, cfg):
    data_args = build_data_args(cfg)
    processor = model.qwen_vl_interface.processor
    processor.tokenizer.model_max_length = int(data_args.model_max_length)
    processor.tokenizer.padding_side = "left"
    processor = update_processor_pixels(processor, data_args)

    dataset_py = str(cfg.datasets.vlm_data.get("dataset_py", "vlm_datasets"))
    if dataset_py == "planner_oft_datasets":
        from starVLA.dataloader.planner_oft_datasets import PlannerOFTThinkDataset

        return PlannerOFTThinkDataset(processor=processor, cfg=cfg, data_args=data_args)
    if dataset_py == "vlm_datasets" and str(cfg.datasets.vlm_data.get("dataformat", "")) in {
        "lerobot_think",
        "astribot_lerobot_think",
    }:
        return LeRobotThinkDataset(processor=processor, cfg=cfg, data_args=data_args)
    raise ValueError(
        f"Unsupported VLM branch dataset: dataset_py={dataset_py!r}, "
        f"dataformat={cfg.datasets.vlm_data.get('dataformat', '')!r}"
    )


def format_prompt_answer(think_dataset, sample: dict):
    formatted = think_dataset._format_answer(sample)
    if len(formatted) == 3:
        prompt, answer, target_instruction = formatted
    else:
        prompt, answer = formatted
        target_instruction = sample.get("subtask_lang") or sample.get("lang")
    return str(prompt), str(answer), str(target_instruction)


def build_generation_inputs(model, images: list, prompt: str, device: str):
    processor = model.qwen_vl_interface.processor
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device), messages


def eos_token_ids(model, device: torch.device) -> torch.Tensor | None:
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    generation_config = model.qwen_vl_interface.model.generation_config
    values = generation_config.eos_token_id
    ids: list[int] = []
    if values is None:
        pass
    elif isinstance(values, int):
        ids.append(int(values))
    else:
        ids.extend(int(item) for item in values)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        ids.append(int(im_end_id))
    ids = sorted(set(ids))
    if not ids:
        return None
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.inference_mode()
def greedy_generate_state_conditioned(model, qwen_inputs: dict, max_new_tokens: int) -> torch.LongTensor:
    qwen_model = model.qwen_vl_interface.model
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    input_ids = qwen_inputs["input_ids"]
    inputs_embeds = qwen_inputs["inputs_embeds"]
    attention_mask = qwen_inputs["attention_mask"]
    device = inputs_embeds.device
    batch_size = int(inputs_embeds.shape[0])
    embed_layer = model._embedding_layer()

    pad_token_id = qwen_model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    pad_token_id = int(pad_token_id)
    stop_ids = eos_token_ids(model, device)

    generation_kwargs = {
        key: value
        for key, value in qwen_inputs.items()
        if key not in {"input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"}
    }
    generated_ids = input_ids
    unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

    for _ in range(int(max_new_tokens)):
        position_ids = model._build_position_ids(generated_ids, attention_mask, qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = qwen_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                **generation_kwargs,
            )
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        if stop_ids is not None:
            next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

        generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )

        if stop_ids is not None:
            is_eos = (next_tokens[:, None] == stop_ids[None, :]).any(dim=-1)
            unfinished = unfinished & ~is_eos
            if int(unfinished.max().item()) == 0:
                break

        next_embeds = embed_layer(next_tokens[:, None]).to(device=device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)

    return generated_ids


def prepare_state_generation_inputs(model, gen_inputs: dict, messages: list, sample: dict):
    state_history = sample.get("state_history", sample.get("state", None))
    if state_history is None:
        raise KeyError("VLM include_state is true, but sample has no state_history/state")
    state_history = np.asarray(state_history, dtype=np.float32)
    if state_history.ndim == 1:
        state_history = state_history[None, :]
    state_insert_index = _qwen_user_content_end_index(messages, model.qwen_vl_interface.processor)

    qwen_inputs = dict(gen_inputs)
    qwen_inputs["state_history"] = [state_history]
    qwen_inputs["state_insert_index"] = torch.tensor(
        [state_insert_index],
        dtype=torch.long,
        device=gen_inputs["input_ids"].device,
    )
    prepared, padded_input_ids = model._prepare_state_conditioned_inputs(
        qwen_inputs,
        [state_history],
        insert_before="supervised",
        insert_indices=qwen_inputs["state_insert_index"],
    )
    prepared["input_ids"] = padded_input_ids
    return prepared, state_insert_index


@torch.inference_mode()
def generate_vlm_text(model, cfg, sample: dict, prompt: str, max_new_tokens: int, device: str):
    images = image_list(sample)
    gen_inputs, messages = build_generation_inputs(model, images, prompt, device)
    include_state = as_bool(cfg.datasets.vlm_data.get("include_state", False))
    state_insert_index = None
    if include_state:
        if not hasattr(model, "_prepare_state_conditioned_inputs"):
            raise ValueError(f"{type(model).__name__} does not support state-conditioned VLM generation")
        gen_inputs, state_insert_index = prepare_state_generation_inputs(model, gen_inputs, messages, sample)
        generated_ids = greedy_generate_state_conditioned(model, gen_inputs, max_new_tokens=max_new_tokens)
        prompt_length = int(gen_inputs["input_ids"].shape[1])
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = model.qwen_vl_interface.model.generate(
                **gen_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )
        prompt_length = int(gen_inputs["input_ids"].shape[1])

    generated_text = model.qwen_vl_interface.processor.tokenizer.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "text": generated_text,
        "token_count": int(generated_ids.shape[1] - prompt_length),
        "state_insert_index": state_insert_index,
    }


def compute_vlm_loss(model, cfg, think_dataset, index: int, device: str) -> float | None:
    instance = think_dataset[index]
    collator = DataCollatorForSupervisedDataset(model.qwen_vl_interface.processor.tokenizer)
    batch = collator([instance])
    batch = move_to_device(batch, device)
    if as_bool(cfg.datasets.vlm_data.get("include_state", False)):
        if not hasattr(model, "prepare_vlm_state_conditioned_inputs"):
            raise ValueError(f"{type(model).__name__} does not support state-conditioned VLM loss")
        batch = model.prepare_vlm_state_conditioned_inputs(batch)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.qwen_vl_interface(**batch, return_dict=True)
    if output.loss is None:
        return None
    return float(output.loss.detach().float().cpu().item())


def compute_action_output(model, sample: dict) -> dict[str, Any] | None:
    if not hasattr(model, "predict_action"):
        return None
    try:
        with torch.inference_mode():
            forward_out = model.forward([sample])
        action_loss = forward_out.get("action_loss", None)
        with torch.inference_mode():
            pred_out = model.predict_action(examples=[sample])
        pred = np.asarray(pred_out["normalized_actions"], dtype=np.float32)
        target = np.asarray(sample["action"], dtype=np.float32)[None, -pred.shape[1] :, :]
        diff = pred - target
        return {
            "status": "ok",
            "action_loss": None if action_loss is None else float(action_loss.detach().float().cpu().item()),
            "prediction_shape": list(pred.shape),
            "target_shape": list(target.shape),
            "l1_mean": float(np.mean(np.abs(diff))),
            "mse_mean": float(np.mean(np.square(diff))),
            "prediction": pred[0].tolist(),
            "target": target[0].tolist(),
        }
    except NotImplementedError:
        return None
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }


def sample_metadata(sample: dict, image_paths: list[str]) -> dict[str, Any]:
    state_history = sample.get("state_history", sample.get("state", None))
    action = sample.get("action", None)
    keys = [
        "lang",
        "task_lang",
        "subtask_lang",
        "history_mode",
        "history_frame_indices",
        "retrieval_indices",
        "retrieval_frame_indices",
        "subtask_keyframe_indices",
        "memory_frame_indices",
        "planner_oft_mode",
        "planner_stride",
        "trajectory_id",
        "base_index",
        "num_frames",
        "num_history_frames",
        "robot_tag",
    ]
    meta = {key: sample[key] for key in keys if key in sample}
    meta["image_files"] = image_paths
    meta["state_history_shape"] = tensor_shape(state_history)
    meta["state_history"] = state_history
    meta["action_shape"] = tensor_shape(action)
    return jsonable(meta)


def write_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)


def export_one_sample(model, cfg, think_dataset, run_name: str, ckpt_path: Path, sample_index: int, sample_dir: Path, args):
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = think_dataset.vla_dataset[sample_index]
    prompt, answer, target_instruction = format_prompt_answer(think_dataset, sample)
    images = image_list(sample)
    image_paths = save_images(images, sample_dir)

    input_payload = {
        "run": run_name,
        "framework": str(cfg.framework.name),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": checkpoint_step(ckpt_path),
        "sample_index": sample_index,
        "prompt": prompt,
        "target_vlm_answer": answer,
        "target_instruction": target_instruction,
        "sample": sample_metadata(sample, image_paths),
    }
    write_json(sample_dir / "input.json", input_payload)

    output_payload: dict[str, Any] = {
        "run": run_name,
        "sample_index": sample_index,
        "status": "ok",
    }
    try:
        output_payload["vlm_loss"] = compute_vlm_loss(model, cfg, think_dataset, sample_index, args.device)
    except Exception as exc:
        output_payload["vlm_loss_error"] = {
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }

    try:
        output_payload["vlm_generation"] = generate_vlm_text(
            model,
            cfg,
            sample,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    except Exception as exc:
        output_payload["status"] = "failed"
        output_payload["vlm_generation_error"] = {
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }

    action_output = compute_action_output(model, sample)
    if action_output is not None:
        output_payload["action_branch"] = action_output

    write_json(sample_dir / "output.json", output_payload)
    return {
        "sample_index": sample_index,
        "sample_dir": str(sample_dir),
        "status": output_payload["status"],
        "vlm_loss": output_payload.get("vlm_loss"),
        "vlm_text": (output_payload.get("vlm_generation") or {}).get("text"),
        "action_l1_mean": (output_payload.get("action_branch") or {}).get("l1_mean"),
    }


def export_run(run_name: str, ckpt_path: Path, args) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "run": run_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": checkpoint_step(ckpt_path),
        "status": "ok",
        "samples": [],
    }
    model = None
    try:
        cfg = load_config(ckpt_path)
        if not has_vlm_branch(cfg):
            result["status"] = "skipped"
            result["reason"] = "no datasets.vlm_data"
            return result

        result["framework"] = str(cfg.framework.name)
        model = load_model(ckpt_path, cfg, args.device)
        think_dataset = build_think_dataset(model, cfg)
        result["dataset_length"] = len(think_dataset)

        run_dir = Path(args.output_root) / run_name
        for sample_index in range(min(int(args.max_samples), len(think_dataset))):
            sample_dir = run_dir / f"sample_{sample_index:03d}"
            result["samples"].append(
                export_one_sample(
                    model,
                    cfg,
                    think_dataset,
                    run_name,
                    ckpt_path,
                    sample_index,
                    sample_dir,
                    args,
                )
            )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=20)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["elapsed_sec"] = round(time.time() - started, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/Checkpoints")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-root", default="results/vlm_branch_samples")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(args.checkpoint_root)
    runs = args.runs or scan_vlm_runs(root)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_root / f"summary_{args.device.replace(':', '')}.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for run_name in runs:
            ckpt = latest_checkpoint(root / run_name)
            if ckpt is None:
                result = {"run": run_name, "status": "skipped", "reason": "no checkpoint"}
            else:
                print(json.dumps({"event": "start", "run": run_name, "checkpoint": str(ckpt)}, ensure_ascii=False), flush=True)
                result = export_run(run_name, ckpt, args)
            results.append(result)
            line = json.dumps(jsonable(result), ensure_ascii=False)
            print(line, flush=True)
            summary_file.write(line + "\n")
            summary_file.flush()

    all_ok = all(item.get("status") in {"ok", "skipped"} for item in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

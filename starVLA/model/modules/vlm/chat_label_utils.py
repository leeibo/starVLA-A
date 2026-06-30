import torch

IGNORE_INDEX = -100


def _prompt_token_length(tokenizer, prompt_text: str) -> int:
    tokenized = tokenizer(prompt_text, add_special_tokens=False)
    return len(tokenized["input_ids"])


def mask_labels_to_response(batch_inputs, tokenizer, prompt_texts, ignore_index: int = IGNORE_INDEX):
    """Keep loss only on assistant response tokens.

    ``prompt_texts`` must be produced from the same chat template as
    ``batch_inputs``, but without the assistant content and with
    ``add_generation_prompt=True``.
    """
    labels = batch_inputs["input_ids"].clone()
    seq_width = labels.size(1)
    attention_mask = batch_inputs.get("attention_mask", None)

    for i, prompt_text in enumerate(prompt_texts):
        prefix_len = _prompt_token_length(tokenizer, prompt_text)
        if attention_mask is not None:
            seq_len = int(attention_mask[i].sum().item())
            left_pad = seq_width - seq_len
            response_start = left_pad + prefix_len
        else:
            response_start = prefix_len

        response_start = max(0, min(response_start, seq_width))
        labels[i, :response_start] = ignore_index

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = ignore_index

    return labels

"""Load Qwen3-4B-Instruct + LoRA adapter for LEQA.

Supports two modes:
  - HuggingFace transformers (for LoRA training via peft)
  - vLLM (for fast batch inference with LoRA)

Default LoRA config: r=16, alpha=16, target_modules="all-linear"
(q/k/v/o/gate/up/down projections), dropout=0.05.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default LoRA hyperparameters per FINAL_PROPOSAL.md section 3.2
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
DEFAULT_BASE_MODEL = "/data1/mq/models/Qwen3-4B-Instruct-2507"


def get_lora_config(
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Create a peft LoraConfig for Qwen3-4B."""
    from peft import LoraConfig, TaskType

    task_map = {"CAUSAL_LM": TaskType.CAUSAL_LM}
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_LORA_TARGET_MODULES,
        task_type=task_map.get(task_type, TaskType.CAUSAL_LM),
        bias="none",
    )


def load_base_model_and_tokenizer(
    base_model_path: str = DEFAULT_BASE_MODEL,
    bf16: bool = True,
    device_map: str | None = "auto",
) -> tuple[Any, Any]:
    """Load Qwen3-4B base model + tokenizer via HuggingFace transformers.

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="bfloat16" if bf16 else "float32",
        device_map=device_map,
        trust_remote_code=True,
    )
    logger.info(
        "Base model loaded: %d params (%.1f M trainable before LoRA)",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )
    return model, tokenizer


def attach_lora(
    model: Any,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> Any:
    """Attach LoRA adapter to a HuggingFace model via peft.

    Returns the PeftModel wrapper.
    """
    from peft import get_peft_model

    lora_cfg = get_lora_config(
        rank=rank, alpha=alpha, dropout=dropout,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA attached: %d trainable / %d total (%.2f%%)",
        trainable, total, 100.0 * trainable / max(total, 1),
    )
    return peft_model


def load_lora_for_inference(
    base_model_path: str = DEFAULT_BASE_MODEL,
    adapter_path: str | None = None,
    bf16: bool = True,
    device_map: str | None = "auto",
) -> tuple[Any, Any]:
    """Load base + merged LoRA adapter for HF inference.

    Returns (model, tokenizer).
    """
    from peft import PeftModel

    model, tokenizer = load_base_model_and_tokenizer(
        base_model_path, bf16=bf16, device_map=device_map,
    )
    if adapter_path is not None:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA adapter loaded and merged")
    return model, tokenizer


def create_vllm_engine(
    base_model_path: str = DEFAULT_BASE_MODEL,
    adapter_path: str | None = None,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 1024,
    seed: int = 42,
) -> Any:
    """Create a vLLM LLM engine with optional LoRA adapter.

    Returns a vllm.LLM instance.
    """
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    kwargs: dict[str, Any] = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "seed": seed,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    if adapter_path is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = DEFAULT_LORA_RANK

    engine = LLM(**kwargs)
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest(
            lora_name="leqa",
            lora_int_id=1,
            lora_path=str(adapter_path),
        )
    return engine, lora_request


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LORA_RANK",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_DROPOUT",
    "DEFAULT_LORA_TARGET_MODULES",
    "get_lora_config",
    "load_base_model_and_tokenizer",
    "attach_lora",
    "load_lora_for_inference",
    "create_vllm_engine",
]

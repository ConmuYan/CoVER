from __future__ import annotations

import os
from typing import Any


def build_gpu_env(visible_devices: str) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": visible_devices}


def resolve_visible_devices(config: dict[str, Any], mode: str = "train") -> str:
    gpu_config = config.get("gpu", {})

    if mode == "train":
        return gpu_config.get("train_visible_devices", "0")
    elif mode == "llm":
        return gpu_config.get("llm_visible_devices", "0")
    elif mode == "both":
        train_dev = gpu_config.get("train_visible_devices", "0")
        llm_dev = gpu_config.get("llm_visible_devices", "0")
        if train_dev == llm_dev:
            return train_dev
        return f"{train_dev},{llm_dev}"
    else:
        return "0"


def print_gpu_info(task_name: str = ""):
    import torch
    prefix = f"[{task_name}] " if task_name else ""
    print(f"{prefix}CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if torch.cuda.is_available():
        print(f"{prefix}torch.cuda.device_count()={torch.cuda.device_count()}")
        print(f"{prefix}torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    else:
        print(f"{prefix}CUDA not available")

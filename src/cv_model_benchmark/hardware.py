from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any


def _run_nvidia_smi() -> list[dict[str, str]]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, memory_total_mb, driver_version, power_limit_w = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": memory_total_mb,
                "driver_version": driver_version,
                "power_limit_w": power_limit_w,
            }
        )
    return gpus


def collect_hardware() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": _run_nvidia_smi(),
    }

    try:
        import torch
    except ImportError:
        data["torch"] = {"available": False}
        return data

    cuda_available = torch.cuda.is_available()
    data["torch"] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
    }
    if cuda_available:
        data["torch"]["device_count"] = torch.cuda.device_count()
        data["torch"]["device_name"] = torch.cuda.get_device_name(0)
        data["torch"]["device_memory_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2)
        )
    return data


def hardware_json(indent: int = 2) -> str:
    return json.dumps(collect_hardware(), indent=indent, sort_keys=True)

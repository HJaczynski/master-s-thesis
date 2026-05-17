from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import load_yaml

RESULT_COLUMNS = [
    "status",
    "run_id",
    "model_id",
    "family",
    "variant_id",
    "precision",
    "model_priority",
    "variant_priority",
    "batch_size",
    "dataset_split",
    "num_images",
    "coco_map_50_95",
    "coco_ap_50",
    "coco_ap_small",
    "coco_ap_medium",
    "coco_ap_large",
    "latency_ms_median",
    "latency_ms_iqr",
    "throughput_images_per_second",
    "params_m",
    "flops_g",
    "model_size_mb",
    "peak_gpu_memory_mb",
    "power_watts",
    "energy_j_per_image",
    "gpu_memory_peak_mb",
    "gpu_utilization_avg_pct",
    "gpu_temperature_max_c",
    "runtime",
    "artifact_path",
    "raw_metrics_path",
    "started_at",
    "ended_at",
    "duration_s",
    "commit",
    "notes",
    "error",
]


def expand_matrix(models_file: str | Path, experiments_file: str | Path) -> list[dict[str, Any]]:
    models_config = load_yaml(models_file)
    experiments_config = load_yaml(experiments_file)
    models = models_config.get("models", [])
    variants = experiments_config.get("variants", [])

    rows: list[dict[str, Any]] = []
    for model in models:
        family = model.get("family")
        for variant in variants:
            applies_to = variant.get("applies_to", [])
            if family not in applies_to and model.get("id") not in applies_to:
                continue
            rows.append(
                {
                    "model_id": model.get("id", ""),
                    "family": family,
                    "model_priority": model.get("priority", ""),
                    "variant_id": variant.get("id", ""),
                    "variant_label": variant.get("label", ""),
                    "variant_priority": variant.get("priority", ""),
                    "precision": variant.get("precision", ""),
                    "input_size": model.get("input_size", ""),
                }
            )
    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model_id",
        "family",
        "variant_id",
        "precision",
        "variant_priority",
        "input_size",
    ]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(output)


def write_results_template(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
    return output_path

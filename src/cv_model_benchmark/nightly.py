from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_yaml
from .gpu_monitor import GpuMonitor
from .hardware import collect_hardware
from .matrix import RESULT_COLUMNS


SUPPORTED_YOLO_VARIANTS = {"fp32_baseline", "fp16", "int8_ptq"}
SUPPORTED_RFDETR_VARIANTS = {"fp32_baseline", "fp16"}
SUPPORTED_MMDETECTION_VARIANTS = {"fp32_baseline"}


def doctor(config_path: str | Path = "configs/nightly.yaml", profile: str = "overnight") -> dict[str, Any]:
    config = load_yaml(config_path)
    profile_config = _profile(config, profile)
    data_yaml = Path(profile_config.get("data_yaml", config.get("data_yaml", "configs/coco_local.yaml")))
    checks = {
        "profile": profile,
        "hardware": collect_hardware(),
        "commands": {
            "nvidia-smi": shutil.which("nvidia-smi"),
            "trtexec": shutil.which("trtexec"),
        },
        "packages": {
            name: importlib.util.find_spec(name) is not None
            for name in [
                "torch",
                "torchvision",
                "ultralytics",
                "onnx",
                "onnxruntime",
                "tensorrt",
                "pynvml",
                "rfdetr",
                "pycocotools",
                "mmcv",
                "mmdet",
                "mmengine",
                "mim",
            ]
        },
        "data_yaml": {
            "path": str(data_yaml),
            "exists": data_yaml.exists(),
        },
    }
    if data_yaml.exists():
        data_config = load_yaml(data_yaml)
        root = (data_yaml.parent / data_config.get("path", "")).resolve()
        val = root / str(data_config.get("val", ""))
        annotations = root / "annotations" / "instances_val2017.json"
        checks["data_yaml"]["dataset_root"] = str(root)
        checks["data_yaml"]["val_exists"] = val.exists()
        checks["data_yaml"]["annotations_path"] = str(annotations)
        checks["data_yaml"]["annotations_exists"] = annotations.exists()
    return checks


def run_nightly(
    config_path: str | Path = "configs/nightly.yaml",
    models_path: str | Path = "configs/models.yaml",
    experiments_path: str | Path = "configs/experiments.yaml",
    profile: str = "overnight",
    dry_run: bool = False,
    limit: int | None = None,
    only_models: list[str] | None = None,
    only_variants: list[str] | None = None,
) -> Path:
    config = load_yaml(config_path)
    model_config = load_yaml(models_path)
    experiment_config = load_yaml(experiments_path)
    profile_config = _profile(config, profile)

    jobs = build_jobs(model_config, experiment_config, config, profile_config, only_models, only_variants)
    if limit is not None:
        jobs = jobs[:limit]

    output_root = Path(profile_config.get("output_dir", config.get("output_dir", "runs/nightly")))
    run_dir = output_root / f"{_timestamp()}_{profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_dir": str(run_dir),
        "profile": profile,
        "created_at": _now(),
        "host": platform.node(),
        "config_path": str(config_path),
        "models_path": str(models_path),
        "experiments_path": str(experiments_path),
        "hardware": collect_hardware(),
        "jobs": jobs,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "doctor.json", doctor(config_path, profile))

    if dry_run:
        _write_json(run_dir / "dry_run_jobs.json", jobs)
        print(f"Dry run wrote {len(jobs)} jobs to {run_dir}")
        return run_dir

    summary_path = run_dir / "summary.csv"
    events_path = run_dir / "events.jsonl"
    _write_csv_header(summary_path)

    failures = 0
    max_failures = int(profile_config.get("max_failures", config.get("max_failures", 20)))
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job['model_id']} / {job['variant_id']} / batch {job['batch_size']}")
        _append_jsonl(events_path, {"event": "started", "time": _now(), **job})
        result = run_job(job, config, profile_config, run_dir)
        _append_result(summary_path, result)
        _append_jsonl(events_path, {"event": "finished", "time": _now(), **result})
        if result["status"] == "failed":
            failures += 1
            if failures >= max_failures:
                _append_jsonl(
                    events_path,
                    {
                        "event": "aborted",
                        "time": _now(),
                        "reason": f"max_failures reached ({max_failures})",
                    },
                )
                break
    print(f"Nightly run finished. Results: {summary_path}")
    return run_dir


def build_jobs(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    config: dict[str, Any],
    profile_config: dict[str, Any],
    only_models: list[str] | None = None,
    only_variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    models = {model["id"]: model for model in model_config.get("models", [])}
    variants = {variant["id"]: variant for variant in experiment_config.get("variants", [])}
    model_ids = only_models or profile_config.get("models", [])
    variant_ids = only_variants or profile_config.get("variants", [])
    batch_sizes = profile_config.get("batch_sizes", [1])
    runtimes = profile_config.get("runtimes", {})
    jobs = []
    for model_id in model_ids:
        model = models.get(model_id)
        if model is None:
            jobs.append(_synthetic_skip_job(model_id, "unknown_model", profile_config))
            continue
        for variant_id in variant_ids:
            variant = variants.get(variant_id)
            if variant is None:
                jobs.append(_synthetic_skip_job(model_id, f"unknown_variant:{variant_id}", profile_config, model))
                continue
            applies_to = variant.get("applies_to", [])
            if model.get("family") not in applies_to and model_id not in applies_to:
                continue
            for batch_size in batch_sizes:
                runtime = runtimes.get(model.get("framework"), {}).get(variant_id)
                runtime = runtime or runtimes.get(model.get("family"), {}).get(variant_id)
                runtime = runtime or _default_runtime(variant_id)
                jobs.append(
                    {
                        "run_id": _run_id(model_id, variant_id, runtime, batch_size),
                        "model_id": model_id,
                        "family": model.get("family"),
                        "framework": model.get("framework"),
                        "variant_id": variant_id,
                        "precision": variant.get("precision"),
                        "model_priority": model.get("priority"),
                        "variant_priority": variant.get("priority"),
                        "batch_size": int(batch_size),
                        "runtime": runtime,
                        "model": model,
                        "variant": variant,
                    }
                )
    return jobs


def run_job(job: dict[str, Any], config: dict[str, Any], profile_config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    started_at = _now()
    start = time.perf_counter()
    base = _base_result(job, started_at)
    try:
        if job.get("framework") == "ultralytics":
            result = _run_yolo_job(job, config, profile_config, run_dir)
        elif job.get("framework") == "rfdetr":
            result = _run_rfdetr_job(job, config, profile_config, run_dir)
        elif job.get("framework") == "mmdetection":
            result = _run_mmdetection_job(job, config, profile_config, run_dir)
        else:
            result = {
                "status": "skipped",
                "notes": f"Adapter for framework '{job.get('framework')}' is not implemented in the overnight runner yet.",
            }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        raw_path = run_dir / "raw" / f"{job.get('run_id', 'unknown')}_error.json"
        _write_json(raw_path, {"error": error_text, "traceback": traceback.format_exc()})
        result = {
            "status": "failed",
            "error": error_text,
            "notes": f"Experiment failed; full traceback is in {raw_path}.",
            "raw_metrics_path": str(raw_path),
        }
    ended_at = _now()
    base.update(result)
    base["started_at"] = started_at
    base["ended_at"] = ended_at
    base["duration_s"] = round(time.perf_counter() - start, 3)
    return _normalize_row(base)


def _run_rfdetr_job(
    job: dict[str, Any],
    config: dict[str, Any],
    profile_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    if job["variant_id"] not in SUPPORTED_RFDETR_VARIANTS:
        return {
            "status": "skipped",
            "notes": (
                f"RF-DETR variant '{job['variant_id']}' is queued for the thesis matrix, "
                "but needs the token-merge/ONNX/TensorRT implementation before it can run unattended."
            ),
        }
    if importlib.util.find_spec("rfdetr") is None:
        return {
            "status": "skipped",
            "notes": "rfdetr is not installed. Install the detector stack before running RF-DETR jobs.",
        }
    if importlib.util.find_spec("pycocotools") is None:
        return {
            "status": "skipped",
            "notes": "pycocotools is not installed, so RF-DETR COCO evaluation cannot run.",
        }

    coco_paths = _resolve_coco_paths(config, profile_config)
    missing = [label for label, path in coco_paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{label}={coco_paths[label]}" for label in missing)
        return {
            "status": "skipped",
            "notes": f"COCO validation data is missing for RF-DETR: {missing_text}. Run scripts/download_coco_val.ps1.",
        }

    model_info = job["model"]
    checkpoint = _resolve_model_checkpoint(model_info.get("checkpoint"))
    if checkpoint is not None and not checkpoint.exists():
        return {
            "status": "skipped",
            "notes": f"RF-DETR checkpoint is missing: {checkpoint}. Update configs/models.yaml or download the weights.",
        }

    import torch

    rfdetr_module = importlib.import_module("rfdetr")
    model_cls = getattr(rfdetr_module, str(model_info.get("class_name", "RFDETRMedium")))
    device = _rfdetr_device(profile_config.get("device", config.get("device", 0)), torch)
    batch = int(job["batch_size"])
    imgsz = int(model_info.get("input_size", 576))
    dtype = torch.float16 if job["variant_id"] == "fp16" else torch.float32
    constructor_kwargs: dict[str, Any] = {"device": device, "resolution": imgsz}
    if checkpoint is not None:
        constructor_kwargs["pretrain_weights"] = str(checkpoint)

    detector = model_cls(**constructor_kwargs)
    rfdetr_cfg = config.get("rfdetr", {})
    if rfdetr_cfg.get("optimize_for_inference", True):
        detector.optimize_for_inference(
            compile=bool(rfdetr_cfg.get("compile", False)),
            batch_size=batch,
            dtype=dtype,
        )

    records = _load_coco_image_records(
        coco_paths["annotations"],
        coco_paths["images"],
        profile_config.get("max_images"),
    )
    threshold = float(rfdetr_cfg.get("score_threshold", 0.001))
    predictions: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    image_ids = [int(record["id"]) for record in records]

    with GpuMonitor(device_index=_device_index(profile_config.get("device", config.get("device", 0))), interval_s=float(config.get("gpu_sample_interval_s", 1.0))) as monitor:
        for chunk in _chunks(records, batch):
            images = [_load_rgb_image(record["path"]) for record in chunk]
            _sync_cuda(torch, device)
            started = time.perf_counter()
            detections = detector.predict(images if len(images) > 1 else images[0], threshold=threshold, shape=(imgsz, imgsz))
            _sync_cuda(torch, device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latency_ms.extend([elapsed_ms / len(chunk)] * len(chunk))
            if not isinstance(detections, list):
                detections = [detections]
            for record, det in zip(chunk, detections):
                predictions.extend(_detections_to_coco_results(int(record["id"]), det))
    telemetry = monitor.summary(num_images=len(records))

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = raw_dir / f"{job['run_id']}_predictions.json"
    _write_json(predictions_path, predictions)
    coco_metrics = _evaluate_coco_bbox(coco_paths["annotations"], predictions, image_ids)
    latency = _latency_metrics(latency_ms)
    raw_path = raw_dir / f"{job['run_id']}.json"
    _write_json(
        raw_path,
        {
            "coco": coco_metrics,
            "latency_ms": latency_ms,
            "num_predictions": len(predictions),
            "predictions_path": str(predictions_path),
            "telemetry": telemetry,
        },
    )

    model_size_mb = None
    if checkpoint is not None and checkpoint.exists():
        model_size_mb = checkpoint.stat().st_size / (1024**2)

    row = {
        "status": "completed",
        "num_images": len(records),
        "coco_map_50_95": coco_metrics.get("coco_map_50_95"),
        "coco_ap_50": coco_metrics.get("coco_ap_50"),
        "coco_ap_small": coco_metrics.get("coco_ap_small"),
        "coco_ap_medium": coco_metrics.get("coco_ap_medium"),
        "coco_ap_large": coco_metrics.get("coco_ap_large"),
        "latency_ms_median": latency.get("median"),
        "latency_ms_iqr": latency.get("iqr"),
        "throughput_images_per_second": latency.get("throughput_images_per_second"),
        "params_m": model_info.get("published_params_m"),
        "flops_g": model_info.get("published_flops_g"),
        "model_size_mb": model_size_mb,
        "artifact_path": str(checkpoint) if checkpoint is not None else str(model_info.get("checkpoint")),
        "raw_metrics_path": str(raw_path),
        "power_watts": telemetry.get("gpu_power_avg_watts"),
        "energy_j_per_image": telemetry.get("gpu_energy_j_per_image"),
        "gpu_memory_peak_mb": telemetry.get("gpu_memory_peak_mb"),
        "gpu_utilization_avg_pct": telemetry.get("gpu_utilization_avg_pct"),
        "gpu_temperature_max_c": telemetry.get("gpu_temperature_max_c"),
        "notes": f"RF-DETR PyTorch {job['variant_id']}; threshold={threshold}; telemetry={telemetry.get('gpu_monitor_backend')}",
    }
    return row


def _run_mmdetection_job(
    job: dict[str, Any],
    config: dict[str, Any],
    profile_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    if job["variant_id"] not in SUPPORTED_MMDETECTION_VARIANTS:
        return {
            "status": "skipped",
            "notes": (
                f"Swin/MMDetection variant '{job['variant_id']}' is present in the queue, "
                "but this runner currently launches the FP32 MMDetection baseline only."
            ),
        }
    missing_packages = [
        name
        for name in ["mim", "mmdet", "mmengine"]
        if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        return {
            "status": "skipped",
            "notes": (
                "Swin/MMDetection job is scheduled, but the MMDetection stack is missing: "
                f"{', '.join(missing_packages)}. Run scripts/install_swin_mmdetection.ps1 after the base stack."
            ),
        }

    model_info = job["model"]
    config_path = _resolve_path(model_info.get("config"))
    checkpoint_path = _resolve_path(model_info.get("checkpoint"))
    missing_paths = []
    if config_path is None or not config_path.exists():
        missing_paths.append(f"config={config_path}")
    if checkpoint_path is None or not checkpoint_path.exists():
        missing_paths.append(f"checkpoint={checkpoint_path}")
    if missing_paths:
        return {
            "status": "skipped",
            "notes": (
                "Swin/MMDetection is in the overnight queue, but its local assets are missing: "
                f"{', '.join(missing_paths)}. Put the official config/checkpoint at the paths in configs/models.yaml."
            ),
        }

    work_dir = run_dir / "mmdetection" / job["run_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "mim_test.log"
    mim_exe = shutil.which("mim") or "mim"
    command = [
        mim_exe,
        "test",
        "mmdet",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--work-dir",
        str(work_dir),
    ]
    timeout_s = int(profile_config.get("command_timeout_s", config.get("command_timeout_s", 24 * 60 * 60)))
    with GpuMonitor(device_index=_device_index(profile_config.get("device", config.get("device", 0))), interval_s=float(config.get("gpu_sample_interval_s", 1.0))) as monitor:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    log_path.write_text(completed.stdout + "\n\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    telemetry = monitor.summary()
    parsed = _parse_mmdetection_metrics(completed.stdout + "\n" + completed.stderr)

    raw_path = run_dir / "raw" / f"{job['run_id']}.json"
    _write_json(
        raw_path,
        {
            "command": command,
            "returncode": completed.returncode,
            "log_path": str(log_path),
            "parsed_metrics": parsed,
            "telemetry": telemetry,
        },
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "raw_metrics_path": str(raw_path),
            "error": f"mim test exited with code {completed.returncode}",
            "notes": f"See {log_path}",
        }

    return {
        "status": "completed",
        "num_images": profile_config.get("max_images"),
        "coco_map_50_95": parsed.get("coco_map_50_95"),
        "coco_ap_50": parsed.get("coco_ap_50"),
        "params_m": model_info.get("published_params_m"),
        "flops_g": model_info.get("published_flops_g"),
        "artifact_path": str(checkpoint_path),
        "raw_metrics_path": str(raw_path),
        "power_watts": telemetry.get("gpu_power_avg_watts"),
        "energy_j_per_image": telemetry.get("gpu_energy_j_per_image"),
        "gpu_memory_peak_mb": telemetry.get("gpu_memory_peak_mb"),
        "gpu_utilization_avg_pct": telemetry.get("gpu_utilization_avg_pct"),
        "gpu_temperature_max_c": telemetry.get("gpu_temperature_max_c"),
        "notes": f"MMDetection/Swin via mim; log={log_path}; telemetry={telemetry.get('gpu_monitor_backend')}",
    }


def _run_yolo_job(
    job: dict[str, Any],
    config: dict[str, Any],
    profile_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    if job["variant_id"] not in SUPPORTED_YOLO_VARIANTS:
        return {
            "status": "skipped",
            "notes": f"Variant '{job['variant_id']}' needs a custom implementation and was not scheduled for overnight automation.",
        }
    if importlib.util.find_spec("ultralytics") is None:
        return {
            "status": "skipped",
            "notes": "ultralytics is not installed. Run scripts/install_nvidia_stack.ps1 first.",
        }

    from ultralytics import YOLO

    model_info = job["model"]
    variant_id = job["variant_id"]
    runtime = job["runtime"]
    data_yaml = _resolve_data_yaml(config, profile_config)
    label_note = _ensure_yolo_coco_labels(data_yaml)
    max_images = profile_config.get("max_images")
    run_data_yaml = _prepare_yolo_subset_yaml(data_yaml, max_images, run_dir, job, "val")
    calibration_images = _effective_calibration_images(config, profile_config)
    if max_images:
        calibration_images = min(int(max_images), calibration_images)
    calibration_data_yaml = _prepare_yolo_subset_yaml(data_yaml, calibration_images, run_dir, job, "calib")
    imgsz = int(model_info.get("input_size", 640))
    batch = int(job["batch_size"])
    device = str(profile_config.get("device", config.get("device", 0)))

    artifact_path: str | None = None
    weights = model_info.get("weights")
    if runtime == "tensorrt":
        artifact_path = _ensure_yolo_tensorrt_engine(
            YOLO,
            weights,
            job,
            config,
            profile_config,
            run_dir,
            calibration_data_yaml,
            imgsz,
            batch,
            device,
            calibration_images,
        )
        yolo_model = YOLO(artifact_path)
    else:
        yolo_model = YOLO(weights)

    val_kwargs: dict[str, Any] = {
        "data": run_data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str((run_dir / "ultralytics").resolve()),
        "name": job["run_id"],
        "exist_ok": True,
        "plots": False,
        "verbose": False,
    }
    if runtime == "pytorch" and variant_id == "fp16":
        val_kwargs["half"] = True

    with GpuMonitor(device_index=int(device), interval_s=float(config.get("gpu_sample_interval_s", 1.0))) as monitor:
        metrics = yolo_model.val(**val_kwargs)
    num_images = int(max_images or profile_config.get("dataset_size", config.get("dataset_size", 5000)))
    telemetry = monitor.summary(num_images=num_images)
    raw_path = run_dir / "raw" / f"{job['run_id']}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(raw_path, _extract_ultralytics_raw(metrics))

    row = _extract_ultralytics_result(metrics)
    row.update(
        {
            "status": "completed",
            "num_images": num_images,
            "artifact_path": artifact_path or str(weights),
            "raw_metrics_path": str(raw_path),
            "params_m": model_info.get("published_params_m"),
            "flops_g": model_info.get("published_flops_g"),
            "power_watts": telemetry.get("gpu_power_avg_watts"),
            "energy_j_per_image": telemetry.get("gpu_energy_j_per_image"),
            "gpu_memory_peak_mb": telemetry.get("gpu_memory_peak_mb"),
            "gpu_utilization_avg_pct": telemetry.get("gpu_utilization_avg_pct"),
            "gpu_temperature_max_c": telemetry.get("gpu_temperature_max_c"),
            "notes": "; ".join(
                note
                for note in [
                    f"Ultralytics {runtime}",
                    label_note,
                    f"telemetry={telemetry.get('gpu_monitor_backend')}",
                ]
                if note
            ),
        }
    )
    return row


def _ensure_yolo_tensorrt_engine(
    yolo_cls: Any,
    weights: str,
    job: dict[str, Any],
    config: dict[str, Any],
    profile_config: dict[str, Any],
    run_dir: Path,
    data_yaml: str,
    imgsz: int,
    batch: int,
    device: str,
    calibration_images: int,
) -> str:
    export_cfg = config.get("tensorrt", {})
    precision = "int8" if job["variant_id"] == "int8_ptq" else "fp16"
    data_yaml = _absolute_if_local(data_yaml)
    export_dir = Path(export_cfg.get("export_dir", "exports/tensorrt"))
    if not export_dir.is_absolute():
        export_dir = (Path.cwd() / export_dir).resolve()
    calibration_tag = f"_calib{calibration_images}" if precision == "int8" else ""
    export_dir = export_dir / job["model_id"] / precision / f"imgsz{imgsz}_b{batch}{calibration_tag}"
    export_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(export_dir.glob("*.engine"))
    if existing and profile_config.get("resume", True):
        return str(existing[-1])

    export_weights = _prepare_yolo_export_weights(weights, export_dir)

    old_cwd = Path.cwd()
    os.chdir(export_dir)
    try:
        model = yolo_cls(export_weights)
        export_kwargs: dict[str, Any] = {
            "format": "engine",
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workspace": export_cfg.get("workspace_gb", 4),
            "dynamic": bool(export_cfg.get("dynamic", False)),
            "verbose": bool(export_cfg.get("verbose", False)),
        }
        if precision == "int8":
            export_kwargs["int8"] = True
            export_kwargs["data"] = data_yaml
            export_kwargs["fraction"] = float(export_cfg.get("calibration_fraction", 1.0))
        else:
            export_kwargs["half"] = True
        exported = model.export(**export_kwargs)
    finally:
        os.chdir(old_cwd)

    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = export_dir / exported_path
    if not exported_path.exists():
        engines = sorted(export_dir.glob("*.engine"))
        if engines:
            exported_path = engines[-1]
    return str(exported_path)


def _extract_ultralytics_result(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    inference_ms = _as_float(speed.get("inference"))
    postprocess_ms = _as_float(speed.get("postprocess"))
    preprocess_ms = _as_float(speed.get("preprocess"))
    latency_ms = None
    if inference_ms is not None:
        latency_ms = inference_ms + (postprocess_ms or 0.0)
    return {
        "coco_map_50_95": _as_float(getattr(box, "map", None)),
        "coco_ap_50": _as_float(getattr(box, "map50", None)),
        "coco_ap_small": None,
        "coco_ap_medium": None,
        "coco_ap_large": None,
        "latency_ms_median": latency_ms,
        "latency_ms_iqr": None,
        "throughput_images_per_second": 1000.0 / latency_ms if latency_ms else None,
        "notes": f"speed_ms preprocess={preprocess_ms}, inference={inference_ms}, postprocess={postprocess_ms}",
    }


def _extract_ultralytics_raw(metrics: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in ["results_dict", "speed", "task", "save_dir"]:
        value = getattr(metrics, name, None)
        data[name] = str(value) if isinstance(value, Path) else value
    box = getattr(metrics, "box", None)
    if box is not None:
        data["box"] = {
            "map": _as_float(getattr(box, "map", None)),
            "map50": _as_float(getattr(box, "map50", None)),
            "map75": _as_float(getattr(box, "map75", None)),
        }
    return data


def _ensure_yolo_coco_labels(data_yaml: str) -> str | None:
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists() or data_yaml_path.name == "coco.yaml":
        return None

    data_config = load_yaml(data_yaml_path)
    root = Path(str(data_config.get("path", "")))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    val_dir = root / str(data_config.get("val", "val2017"))
    annotations = root / str(data_config.get("annotations", "annotations/instances_val2017.json"))
    if not val_dir.exists() or not annotations.exists():
        return None

    annotation_data = json.loads(annotations.read_text(encoding="utf-8"))
    images = annotation_data.get("images", [])
    if not images:
        return None

    expected_labels = [val_dir / f"{Path(image['file_name']).stem}.txt" for image in images]
    if all(path.exists() for path in expected_labels):
        return "YOLO labels ready"

    image_by_id = {int(image["id"]): image for image in images}
    label_lines: dict[str, list[str]] = {str(image["file_name"]): [] for image in images}
    categories = sorted(annotation_data.get("categories", []), key=lambda item: int(item["id"]))
    cat_to_label = {int(category["id"]): index for index, category in enumerate(categories)}

    for annotation in annotation_data.get("annotations", []):
        if annotation.get("iscrowd"):
            continue
        image = image_by_id.get(int(annotation.get("image_id", -1)))
        if image is None:
            continue
        category_id = int(annotation.get("category_id", -1))
        if category_id not in cat_to_label:
            continue
        x, y, width, height = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
        if width <= 0 or height <= 0:
            continue
        image_width = float(image["width"])
        image_height = float(image["height"])
        x_center = (x + width / 2.0) / image_width
        y_center = (y + height / 2.0) / image_height
        label_lines[str(image["file_name"])].append(
            "{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}".format(
                label=cat_to_label[category_id],
                x=_clip01(x_center),
                y=_clip01(y_center),
                w=_clip01(width / image_width),
                h=_clip01(height / image_height),
            )
        )

    for image in images:
        label_path = val_dir / f"{Path(image['file_name']).stem}.txt"
        label_path.write_text("\n".join(label_lines[str(image["file_name"])]) + "\n", encoding="utf-8")

    cache_path = val_dir.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()
    return f"generated YOLO labels from {annotations.name}"


def _prepare_yolo_subset_yaml(
    data_yaml: str,
    max_images: int | None,
    run_dir: Path,
    job: dict[str, Any],
    suffix: str,
) -> str:
    if not max_images:
        return data_yaml

    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists() or data_yaml_path.name == "coco.yaml":
        return data_yaml

    data_config = load_yaml(data_yaml_path)
    root = Path(str(data_config.get("path", "")))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    val_dir = root / str(data_config.get("val", "val2017"))
    annotations = root / str(data_config.get("annotations", "annotations/instances_val2017.json"))
    if not val_dir.exists() or not annotations.exists():
        return data_yaml

    annotation_data = json.loads(annotations.read_text(encoding="utf-8"))
    images = sorted(annotation_data.get("images", []), key=lambda item: int(item["id"]))[: int(max_images)]
    subset_dir = run_dir / "datasets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    image_list_path = subset_dir / f"{job['run_id']}_{suffix}.txt"
    image_list_path.write_text(
        "\n".join(str((val_dir / image["file_name"]).resolve()) for image in images) + "\n",
        encoding="utf-8",
    )

    subset_config = dict(data_config)
    subset_config["path"] = str(root)
    subset_config["val"] = str(image_list_path.resolve())
    subset_yaml_path = subset_dir / f"{job['run_id']}_{suffix}_data.yaml"
    import yaml

    subset_yaml_path.write_text(yaml.safe_dump(subset_config, sort_keys=False), encoding="utf-8")
    return str(subset_yaml_path.resolve())


def _load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _resolve_coco_paths(config: dict[str, Any], profile_config: dict[str, Any]) -> dict[str, Path]:
    data_yaml = Path(_resolve_data_yaml(config, profile_config))
    if not data_yaml.exists():
        return {
            "data_yaml": data_yaml,
            "images": data_yaml.parent / "missing-val2017",
            "annotations": data_yaml.parent / "missing-instances_val2017.json",
        }
    data_config = load_yaml(data_yaml)
    root = (data_yaml.parent / str(data_config.get("path", ""))).resolve()
    images = root / str(data_config.get("val", "val2017"))
    annotations = root / str(data_config.get("annotations", "annotations/instances_val2017.json"))
    return {"data_yaml": data_yaml, "images": images, "annotations": annotations}


def _load_coco_image_records(annotations_path: Path, images_dir: Path, max_images: int | None) -> list[dict[str, Any]]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = sorted(annotations.get("images", []), key=lambda item: int(item["id"]))
    if max_images:
        images = images[: int(max_images)]
    return [
        {
            "id": int(image["id"]),
            "file_name": image["file_name"],
            "path": images_dir / image["file_name"],
        }
        for image in images
    ]


def _detections_to_coco_results(image_id: int, detections: Any) -> list[dict[str, Any]]:
    xyxy = getattr(detections, "xyxy", None)
    confidence = getattr(detections, "confidence", None)
    class_id = getattr(detections, "class_id", None)
    if xyxy is None or confidence is None or class_id is None:
        return []

    category_ids = _coco_category_ids()
    results: list[dict[str, Any]] = []
    for box, score, label in zip(xyxy, confidence, class_id):
        category_id = _resolve_coco_category_id(int(label), category_ids)
        if category_id is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        results.append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x1, y1, width, height],
                "score": float(score),
            }
        )
    return results


def _coco_category_ids() -> list[int]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        return sorted(int(category_id) for category_id in COCO_CLASSES.keys())
    except Exception:
        return [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ]


def _resolve_coco_category_id(label: int, category_ids: list[int]) -> int | None:
    if label in category_ids:
        return label
    if 0 <= label < len(category_ids):
        return category_ids[label]
    return None


def _evaluate_coco_bbox(
    annotations_path: Path,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
) -> dict[str, float | None]:
    if not predictions:
        return {
            "coco_map_50_95": 0.0,
            "coco_ap_50": 0.0,
            "coco_ap_small": 0.0,
            "coco_ap_medium": 0.0,
            "coco_ap_large": 0.0,
            "coco_stats": [0.0] * 12,
        }

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            coco_gt = COCO(str(annotations_path))
            coco_dt = coco_gt.loadRes(predictions)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    return {
        "coco_map_50_95": stats[0],
        "coco_ap_50": stats[1],
        "coco_ap_small": stats[3],
        "coco_ap_medium": stats[4],
        "coco_ap_large": stats[5],
        "coco_stats": stats,
    }


def _latency_metrics(latencies_ms: list[float]) -> dict[str, float | None]:
    if not latencies_ms:
        return {"median": None, "iqr": None, "throughput_images_per_second": None}
    median = statistics.median(latencies_ms)
    iqr = None
    if len(latencies_ms) >= 4:
        quartiles = statistics.quantiles(latencies_ms, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    return {
        "median": median,
        "iqr": iqr,
        "throughput_images_per_second": 1000.0 / median if median else None,
    }


def _parse_mmdetection_metrics(text: str) -> dict[str, float | None]:
    import re

    parsed = {
        "coco_map_50_95": None,
        "coco_ap_50": None,
    }
    patterns = {
        "coco_map_50_95": [
            r"(?:coco/)?bbox_mAP\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"bbox_mAP_copypaste\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
        "coco_ap_50": [
            r"(?:coco/)?bbox_mAP_50\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"bbox_mAP_copypaste\s*[:=]\s*[0-9]*\.?[0-9]+\s+([0-9]*\.?[0-9]+)",
        ],
    }
    for key, key_patterns in patterns.items():
        for pattern in key_patterns:
            match = re.search(pattern, text)
            if match:
                parsed[key] = float(match.group(1))
                break
    return parsed


def _resolve_model_checkpoint(value: Any) -> Path | None:
    if value is None or str(value).strip().lower() == "auto":
        return None
    return _resolve_path(value)


def _resolve_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _chunks(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _rfdetr_device(value: Any, torch_module: Any) -> str:
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}" if torch_module.cuda.is_available() else "cpu"
    return text


def _device_index(value: Any) -> int:
    text = str(value)
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    return 0


def _sync_cuda(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _resolve_data_yaml(config: dict[str, Any], profile_config: dict[str, Any]) -> str:
    data_yaml = Path(profile_config.get("data_yaml", config.get("data_yaml", "configs/coco_local.yaml")))
    if data_yaml.exists():
        return str(data_yaml.resolve())
    if profile_config.get("allow_ultralytics_download", config.get("allow_ultralytics_download", False)):
        return "coco.yaml"
    return str(data_yaml)


def _absolute_if_local(value: str) -> str:
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    return value


def _prepare_yolo_export_weights(weights: str, export_dir: Path) -> str:
    source = Path(weights)
    if source.exists():
        source = source.resolve()
        target = export_dir / source.name
        if source != target:
            shutil.copy2(source, target)
        return str(target)
    return weights


def _effective_calibration_images(config: dict[str, Any], profile_config: dict[str, Any]) -> int:
    export_cfg = config.get("tensorrt", {})
    configured = export_cfg.get("calibration_images")
    if configured:
        return int(configured)
    dataset_size = int(profile_config.get("dataset_size", config.get("dataset_size", 5000)))
    fraction = float(export_cfg.get("calibration_fraction", 1.0))
    return max(1, int(dataset_size * fraction))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown profile '{profile}'. Available profiles: {', '.join(sorted(profiles))}")
    merged = dict(config.get("defaults", {}))
    merged.update(profiles[profile] or {})
    return merged


def _default_runtime(variant_id: str) -> str:
    if variant_id == "int8_ptq":
        return "tensorrt"
    return "pytorch"


def _fraction(max_images: int | None, dataset_size: int | None) -> float | None:
    if not max_images or not dataset_size:
        return None
    return max(0.001, min(1.0, float(max_images) / float(dataset_size)))


def _base_result(job: dict[str, Any], started_at: str) -> dict[str, Any]:
    return {
        "status": "started",
        "run_id": job["run_id"],
        "model_id": job["model_id"],
        "family": job.get("family"),
        "variant_id": job["variant_id"],
        "precision": job.get("precision"),
        "model_priority": job.get("model_priority"),
        "variant_priority": job.get("variant_priority"),
        "batch_size": job["batch_size"],
        "dataset_split": "val2017",
        "runtime": job.get("runtime"),
        "started_at": started_at,
        "commit": _git_commit(),
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: _stringify(row.get(column)) for column in RESULT_COLUMNS}


def _synthetic_skip_job(
    model_id: str,
    reason: str,
    profile_config: dict[str, Any],
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": _run_id(model_id, reason, "skip", 0),
        "model_id": model_id,
        "family": model.get("family") if model else None,
        "framework": model.get("framework") if model else None,
        "variant_id": reason,
        "precision": None,
        "model_priority": model.get("priority") if model else None,
        "variant_priority": None,
        "batch_size": profile_config.get("batch_sizes", [1])[0],
        "runtime": "skip",
        "model": model or {},
        "variant": {},
    }


def _run_id(model_id: str, variant_id: str, runtime: str, batch_size: int) -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"{model_id}__{variant_id}__{runtime}__b{batch_size}__{suffix}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        return completed.stdout.strip()
    except Exception:
        return None


def _write_csv_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()


def _append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writerow(row)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, default=_json_default, sort_keys=True) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default, sort_keys=True), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return value

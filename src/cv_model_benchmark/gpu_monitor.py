from __future__ import annotations

import statistics
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass
class GpuSample:
    timestamp_s: float
    power_watts: float | None = None
    memory_used_mb: float | None = None
    utilization_pct: float | None = None
    temperature_c: float | None = None


class GpuMonitor:
    """Sample NVIDIA GPU telemetry during an experiment.

    NVML via nvidia-ml-py is preferred. If it is not installed, the monitor falls
    back to nvidia-smi. Missing telemetry should not fail an overnight run.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 1.0) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self.backend = "none"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        self._setup_backend()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        if self._nvml is not None:
            with suppress(Exception):
                self._nvml.nvmlShutdown()

    def summary(self, num_images: int | None = None) -> dict[str, Any]:
        powers = [s.power_watts for s in self.samples if s.power_watts is not None]
        memories = [s.memory_used_mb for s in self.samples if s.memory_used_mb is not None]
        utils = [s.utilization_pct for s in self.samples if s.utilization_pct is not None]
        temps = [s.temperature_c for s in self.samples if s.temperature_c is not None]
        duration_s = 0.0
        if len(self.samples) >= 2:
            duration_s = self.samples[-1].timestamp_s - self.samples[0].timestamp_s
        avg_power = statistics.fmean(powers) if powers else None
        energy_j = avg_power * duration_s if avg_power is not None else None
        return {
            "gpu_monitor_backend": self.backend,
            "gpu_sample_count": len(self.samples),
            "gpu_power_avg_watts": avg_power,
            "gpu_power_max_watts": max(powers) if powers else None,
            "gpu_energy_j": energy_j,
            "gpu_energy_j_per_image": energy_j / num_images if energy_j is not None and num_images else None,
            "gpu_memory_peak_mb": max(memories) if memories else None,
            "gpu_utilization_avg_pct": statistics.fmean(utils) if utils else None,
            "gpu_temperature_max_c": max(temps) if temps else None,
        }

    def _setup_backend(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.backend = "nvml"
            return
        except Exception:
            self._nvml = None
            self._handle = None

        try:
            subprocess.run(["nvidia-smi", "--help"], capture_output=True, check=True, text=True, timeout=5)
            self.backend = "nvidia-smi"
        except Exception:
            self.backend = "none"

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._sample_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> GpuSample | None:
        now = time.perf_counter()
        if self.backend == "nvml":
            return self._sample_nvml(now)
        if self.backend == "nvidia-smi":
            return self._sample_nvidia_smi(now)
        return None

    def _sample_nvml(self, now: float) -> GpuSample | None:
        assert self._nvml is not None
        assert self._handle is not None
        try:
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            temperature = self._nvml.nvmlDeviceGetTemperature(
                self._handle,
                self._nvml.NVML_TEMPERATURE_GPU,
            )
            power_watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            return GpuSample(
                timestamp_s=now,
                power_watts=power_watts,
                memory_used_mb=memory.used / (1024**2),
                utilization_pct=float(utilization.gpu),
                temperature_c=float(temperature),
            )
        except Exception:
            return None

    def _sample_nvidia_smi(self, now: float) -> GpuSample | None:
        query = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=power.draw,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(query, capture_output=True, check=True, text=True, timeout=5)
            values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
            return GpuSample(
                timestamp_s=now,
                power_watts=_to_float(values[0]),
                memory_used_mb=_to_float(values[1]),
                utilization_pct=_to_float(values[2]),
                temperature_c=_to_float(values[3]),
            )
        except Exception:
            return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


from __future__ import annotations

import json
import os
import platform
import sys
import time
import ctypes
from contextlib import contextmanager
from pathlib import Path

import cv2
import torch


class TrackNetProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.timings: dict[str, float] = {}
        self.counts: dict[str, int | float] = {}
        self.segments: list[dict] = []
        self.tile_intervals: list[dict] = []
        self.memory_samples: list[dict] = []
        self.input_device: str | None = None
        self.model_device: str | None = None
        self.input_dtype: str | None = None
        if enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.sample_memory("start")

    def add_time(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.timings[name] = self.timings.get(name, 0.0) + seconds

    def add_count(self, name: str, value: int | float = 1) -> None:
        if self.enabled:
            self.counts[name] = self.counts.get(name, 0) + value

    @contextmanager
    def stage(self, name: str, synchronize_cuda: bool = False):
        if not self.enabled:
            yield
            return
        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.add_time(name, time.perf_counter() - started)

    def observe_inference(self, model, tensor) -> None:
        if not self.enabled or self.input_device is not None:
            return
        parameter = next(model.parameters())
        self.model_device = str(parameter.device)
        self.input_device = str(tensor.device)
        self.input_dtype = str(tensor.dtype).replace("torch.", "")
        if parameter.device != tensor.device:
            raise RuntimeError(f"Model/input device mismatch: {parameter.device} != {tensor.device}")

    def sample_memory(self, label: str) -> None:
        if not self.enabled:
            return
        ram_mb = None
        try:
            import psutil
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        except ImportError:
            if sys.platform == "win32":
                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                get_memory_info = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
                if get_memory_info is None:
                    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
                get_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
                get_memory_info.restype = ctypes.c_int
                if get_memory_info(handle, ctypes.byref(counters), counters.cb):
                    ram_mb = counters.WorkingSetSize / 1024**2
        sample = {"label": label, "ram_mb": ram_mb}
        if torch.cuda.is_available():
            sample.update({
                "vram_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "vram_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            })
        self.memory_samples.append(sample)

    def finalize(self, path: Path, *, device, video: dict, configuration: dict) -> dict:
        total = time.perf_counter() - self.started
        for name in ("dependency_initialization", "overlay_render"):
            self.timings.setdefault(name, 0.0)
        self.timings["total"] = total
        classified = sum(value for key, value in self.timings.items() if key not in {"total", "unclassified"})
        self.timings["unclassified"] = max(0.0, total - classified)
        timed = {key: {"seconds": value, "percent": value / total * 100 if total else 0.0} for key, value in self.timings.items()}
        cuda = torch.cuda.is_available()
        report = {
            "environment": {
                "python_version": sys.version.split()[0], "platform": platform.platform(),
                "pytorch_version": torch.__version__, "opencv_version": cv2.__version__,
                "cpu_core_count": os.cpu_count(), "cuda_available": cuda,
                "cuda_device_count": torch.cuda.device_count(), "cuda_version": torch.version.cuda,
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            },
            "device": {"requested": str(device), "actual_model": self.model_device, "actual_input": self.input_device, "input_dtype": self.input_dtype},
            "video": video, "configuration": configuration, "counts": self.counts,
            "segments": self.segments, "tile_recovery_intervals": self.tile_intervals,
            "timings": timed,
            "memory": {
                "samples": self.memory_samples,
                "peak_ram_mb": max((x["ram_mb"] for x in self.memory_samples if x["ram_mb"] is not None), default=None),
                "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if cuda else None,
                "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2 if cuda else None,
            },
            "throughput": {
                "full_frame_inference_fps": self.counts.get("full_frame_processed_frames", 0) / self.timings.get("full_frame_inference", 1) if self.timings.get("full_frame_inference") else 0.0,
                "overall_processing_fps": video.get("frame_count", 0) / total if total else 0.0,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nPerformance profile")
        for name, item in sorted(timed.items(), key=lambda x: x[1]["seconds"], reverse=True):
            print(f"  {name:28s} {item['seconds']:9.3f}s {item['percent']:6.2f}%")
        print("profile_json =", path)
        return report

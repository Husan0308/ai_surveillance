from __future__ import annotations

import os
import threading

_NVML_LOCK = threading.Lock()
_NVML_READY = False
_NVML_FAILED = False


def _ensure_nvml():
    global _NVML_READY, _NVML_FAILED
    if _NVML_READY:
        return True
    if _NVML_FAILED:
        return False
    with _NVML_LOCK:
        if _NVML_READY:
            return True
        if _NVML_FAILED:
            return False
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_READY = True
            return True
        except Exception:
            _NVML_FAILED = True
            return False


def _gpu_process_metrics(pid: int):
    result = {
        "gpu_index": None,
        "gpu_process_memory_mb": None,
        "gpu_device_memory_used_mb": None,
        "gpu_device_memory_total_mb": None,
        "gpu_utilization_percent": None,
        "gpu_temperature_c": None,
    }
    if not _ensure_nvml():
        return result
    try:
        import pynvml
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = []
            for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    processes.extend(getter(handle) or [])
                except Exception:
                    pass
            used = None
            for proc in processes:
                if int(getattr(proc, "pid", -1)) != int(pid):
                    continue
                value = getattr(proc, "usedGpuMemory", None)
                if value is not None and int(value) >= 0:
                    used = int(value)
                    break
            if used is None:
                continue
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result.update({
                "gpu_index": index,
                "gpu_process_memory_mb": used / (1024.0 * 1024.0),
                "gpu_device_memory_used_mb": int(memory.used) / (1024.0 * 1024.0),
                "gpu_device_memory_total_mb": int(memory.total) / (1024.0 * 1024.0),
                "gpu_utilization_percent": int(utilization.gpu),
            })
            try:
                result["gpu_temperature_c"] = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            return result
    except Exception:
        pass
    return result


def process_metrics(pid: int | None = None):
    """Return cheap, best-effort process metrics without making health fragile."""
    target_pid = int(pid or os.getpid())
    result = {
        "pid": target_pid,
        "rss_mb": None,
        "vms_mb": None,
        "threads": None,
        "fds": None,
        "cpu_percent": None,
    }
    try:
        import psutil
        proc = psutil.Process(target_pid)
        memory = proc.memory_info()
        result.update({
            "rss_mb": memory.rss / (1024.0 * 1024.0),
            "vms_mb": memory.vms / (1024.0 * 1024.0),
            "threads": proc.num_threads(),
            "cpu_percent": proc.cpu_percent(interval=None),
        })
        if hasattr(proc, "num_fds"):
            try:
                result["fds"] = proc.num_fds()
            except Exception:
                pass
    except Exception:
        pass
    result.update(_gpu_process_metrics(target_pid))
    return result

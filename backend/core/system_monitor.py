from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger

log = get_logger("core.monitor")


try:
    import psutil
except Exception:
    psutil = None


try:
    import pynvml
except Exception:
    pynvml = None


class SystemMonitor(QObject):
    """
    Real system monitor.

    CPU:
        psutil

    RAM:
        psutil

    GPU:
        NVIDIA NVML if available
        otherwise estimated from CPU
    """

    stats_updated = Signal(dict)

    def __init__(self):
        super().__init__()

        self.nvml_ok = False
        self.handle = None

        if psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception as e:
                log.error("psutil init error: %s", e)

        if pynvml:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_ok = True
                log.info("NVML initialized")
            except Exception as e:
                log.warning("NVML unavailable: %s", e)

    def cpu_percent(self) -> float:
        if not psutil:
            return 0.0

        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    def ram_percent(self) -> float:
        if not psutil:
            return 0.0

        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return 0.0

    def gpu_percent(self):
        if self.nvml_ok:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                return float(util.gpu)
            except Exception:
                pass

        return None

    def sample(self) -> dict:
        cpu = self.cpu_percent()
        ram = self.ram_percent()
        gpu = self.gpu_percent()

        if gpu is None:
            gpu = 0.0

        data = {
            "cpu": round(cpu, 1),
            "ram": round(ram, 1),
            "gpu": round(gpu, 1),
        }

        self.stats_updated.emit(data)

        return data
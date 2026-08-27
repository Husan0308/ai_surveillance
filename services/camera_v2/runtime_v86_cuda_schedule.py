from __future__ import annotations

import os

from .runtime_v85_nvdcf_relief import PascalNvDCFReliefRuntime


class PascalCudaScheduleABRuntime(PascalNvDCFReliefRuntime):
    """V8.6 Step 4: A/B only DeepStream's CUDA blocking-sync policy.

    V8.5 kept the batch-1 detector, 512x288@8Hz NvDCF tracker and all bbox
    policy unchanged, but featureImgSizeLevel 2 -> 1 did not recover integrated
    TensorRT latency.  DeepStream 7.1 documents that dGPU pipelines set
    cudaDeviceScheduleBlockingSync by default and explicitly recommends trying
    NVDS_DISABLE_CUDADEV_BLOCKINGSYNC=1 when a GPU-bound pipeline does not reach
    close to full GPU utilization.

    This class changes no tracker/detector/bbox parameters.  The launcher sets the
    environment variable before GStreamer/DeepStream pipeline construction so this
    run is a clean scheduling-policy A/B against V8.5.
    """

    def __init__(self) -> None:
        super().__init__()
        self.v86_blocking_sync_disabled = (
            os.environ.get("NVDS_DISABLE_CUDADEV_BLOCKINGSYNC", "0") == "1"
        )
        print(
            "CAMERA_V86_ARCH "
            f"nvds_disable_cuda_blocking_sync={int(self.v86_blocking_sync_disabled)} "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"feature_level={self.v85_feature_level} detector=batch1-v84 "
            "tracker_resolution_unchanged=1 tracker_rate_unchanged=1 "
            "bbox_policy=unchanged detector_policy=unchanged one_change=cuda-schedule",
            flush=True,
        )

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_V86_STATS "
            f"blocking_sync_disabled={int(self.v86_blocking_sync_disabled)} "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches}",
            flush=True,
        )
        return keep


def main() -> int:
    if os.environ.get("NVDS_DISABLE_CUDADEV_BLOCKINGSYNC") != "1":
        raise SystemExit(
            "CAMERA_V86_PREFLIGHT ERROR: NVDS_DISABLE_CUDADEV_BLOCKINGSYNC=1 required"
        )
    return PascalCudaScheduleABRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())

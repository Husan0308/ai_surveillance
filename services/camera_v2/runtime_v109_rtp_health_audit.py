from __future__ import annotations

import threading
from collections import defaultdict

from .runtime_v105_mux_arrival_audit import PascalTrackerMuxArrivalAuditRuntime


class PascalRtpHealthAuditRuntime(PascalTrackerMuxArrivalAuditRuntime):
    """V10.9: diagnostic-only RTP/jitterbuffer audit for all camera sources.

    V10.7 (CAM-02 latency 60->120 ms) and V10.8 (CAM-02 TCP->UDP) did not
    materially remove the CAM-02 source/tracker tail. Return to the V10.5/V10.4
    baseline (TCP, 60 ms) and inspect the RTP manager's own jitterbuffer stats
    before changing mux, NvDCF, decoder, or bbox behavior again.
    """

    def __init__(self) -> None:
        self._v109_lock = threading.RLock()
        self._v109_managers: dict[str, object] = {}
        self._v109_jitterbuffers: dict[str, list[object]] = defaultdict(list)
        self._v109_last_pushed: dict[tuple[str, int], int] = {}
        super().__init__()
        print(
            "CAMERA_V109_ARCH behavior_change=0 baseline=tcp60-v105-v104 "
            "only_change=rtp-jitterbuffer-health-audit measure=pushed,lost,late,duplicates,avg-jitter",
            flush=True,
        )

    def _configure_rtsp_child(self, bin_, sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(bin_, sub_bin, element, camera)
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return
        try:
            element.connect("new-manager", self._v109_on_new_manager, camera.camera_id)
            print(
                f"CAMERA_V109_RTSP_HOOK camera={camera.camera_id} new_manager=1",
                flush=True,
            )
        except Exception as exc:
            print(
                f"CAMERA_V109_RTSP_HOOK camera={camera.camera_id} new_manager=0 error={type(exc).__name__}",
                flush=True,
            )

    def _v109_on_new_manager(self, _rtspsrc, manager, cid: str) -> None:
        with self._v109_lock:
            self._v109_managers[cid] = manager
        try:
            manager.connect("new-jitterbuffer", self._v109_on_new_jitterbuffer, cid)
            hooked = 1
        except Exception:
            hooked = 0
        name = manager.get_name() if manager is not None else "unknown"
        print(
            f"CAMERA_V109_MANAGER camera={cid} name={name} jitterbuffer_hook={hooked}",
            flush=True,
        )

    def _v109_on_new_jitterbuffer(self, _manager, jitterbuffer, session: int, ssrc: int, cid: str) -> None:
        with self._v109_lock:
            rows = self._v109_jitterbuffers[cid]
            if jitterbuffer not in rows:
                rows.append(jitterbuffer)
            index = rows.index(jitterbuffer)
        print(
            f"CAMERA_V109_JB_READY camera={cid} index={index} session={int(session)} ssrc={int(ssrc)}",
            flush=True,
        )

    @staticmethod
    def _v109_stat_int(stats, key: str) -> int:
        try:
            value = stats.get_value(key)
            return int(value) if value is not None else 0
        except Exception:
            return 0

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v109_lock:
            snapshot = {
                cid: list(self._v109_jitterbuffers.get(cid, ()))
                for cid in self.camera_index
            }

        for cid in self.camera_index:
            rows = snapshot.get(cid, [])
            if not rows:
                print(f"CAMERA_V109_RTP camera={cid} status=no-jitterbuffer", flush=True)
                continue
            for index, jitterbuffer in enumerate(rows):
                try:
                    stats = jitterbuffer.get_property("stats")
                    pushed = self._v109_stat_int(stats, "num-pushed")
                    lost = self._v109_stat_int(stats, "num-lost")
                    late = self._v109_stat_int(stats, "num-late")
                    duplicates = self._v109_stat_int(stats, "num-duplicates")
                    avg_jitter_ns = self._v109_stat_int(stats, "avg-jitter")
                    key = (cid, index)
                    previous = self._v109_last_pushed.get(key)
                    delta_pushed = pushed - previous if previous is not None else 0
                    self._v109_last_pushed[key] = pushed
                    denom = max(1, pushed + lost)
                    loss_pct = 100.0 * max(0, lost) / denom
                    late_pct = 100.0 * max(0, late) / max(1, pushed + late)
                    print(
                        "CAMERA_V109_RTP "
                        f"camera={cid} index={index} pushed={pushed} pushed_5s={max(0, delta_pushed)} "
                        f"lost={lost} loss_pct={loss_pct:.4f} late={late} late_pct={late_pct:.4f} "
                        f"duplicates={duplicates} avg_jitter_ms={avg_jitter_ns / 1_000_000.0:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"CAMERA_V109_RTP camera={cid} index={index} status=error error={type(exc).__name__}",
                        flush=True,
                    )
        return keep


def main() -> int:
    return PascalRtpHealthAuditRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())

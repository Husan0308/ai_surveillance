from __future__ import annotations

import os
import time

from .secure import SecureCameraWallV2

RESTART_EXIT_CODE = 75


class CleanCameraWallV2(SecureCameraWallV2):
    """High-quality camera-only baseline for six RTSP feeds.

    Graph:
        nvurisrcbin/NVDEC -> queue(1) -> nvstreammux -> nvmultistreamtiler
        -> caps -> queue(1) -> nveglglessink

    Deliberately absent: detector, appsink/SHM, TensorRT, NvDCF, OSD and Global ID.
    This runtime exists to prove the camera transport/display path first.
    """

    def __init__(self) -> None:
        self._restart_requested = False
        self._restart_reason = ""
        self._source_started_at: dict[str, float] = {}
        self._last_frames: dict[str, int] = {}
        self._last_progress: dict[str, float] = {}
        super().__init__()

        self._stall_s = max(
            8.0,
            float(os.environ.get("CAMERA_V2_CLEAN_STALL_SEC", "12")),
        )
        now = time.monotonic()
        self._last_frames = {cid: int(self.stats[cid].frames) for cid in self.sources}
        self._last_progress = {cid: now for cid in self.sources}

        # Quality-first scaling is safe here because analytics are completely absent.
        self._set_if(self.mux, "interpolation-method", 4)   # GPU Lanczos on dGPU
        self._set_if(self.tiler, "interpolation-method", 4)
        self._set_if(self.mux, "buffer-pool-size", 12)

        for cid, source in self.sources.items():
            self._set_if(source, "rtsp-reconnect-interval", 2)
            self._set_if(source, "rtsp-reconnect-attempts", 3)
            self._set_if(source, "async-handling", True)

        self._audit_clean_graph()
        print(
            "CAMERA_CLEAN_ARCH detector=0 tracker=0 appsink=0 shm=0 tensorrt=0 "
            "osd=0 global_id=0 deepstream=decode/mux/tiler/egl",
            flush=True,
        )
        print(
            "CAMERA_CLEAN_QUALITY "
            f"mux={self.frame_width}x{self.frame_height}/lanczos "
            f"wall={self.wall_width}x{self.wall_height}/lanczos "
            f"tiles={self.tiler_columns}x{self.tiler_rows} "
            f"tile={self.wall_width // self.tiler_columns}x{self.wall_height // self.tiler_rows} "
            "latest_queues=1 sync=0 qos=0",
            flush=True,
        )

    @staticmethod
    def _peer_name(element, pad_name: str) -> str | None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            return None
        peer = pad.get_peer()
        if peer is None:
            return None
        parent = peer.get_parent_element()
        return parent.get_name() if parent is not None else None

    def _expect_peer(self, element, pad_name: str, expected: str, label: str) -> None:
        actual = self._peer_name(element, pad_name)
        if actual != expected:
            raise RuntimeError(
                f"CAMERA_CLEAN_PIPELINE_AUDIT {label}: expected={expected} actual={actual}"
            )

    def _audit_clean_graph(self) -> None:
        self._expect_peer(self.mux, "src", self.tiler.get_name(), "mux->tiler")
        self._expect_peer(self.tiler, "src", self.wall_caps.get_name(), "tiler->wall_geometry")
        self._expect_peer(self.wall_caps, "src", self.wall_queue.get_name(), "wall_geometry->queue")
        self._expect_peer(self.wall_queue, "src", self.sink.get_name(), "queue->egl")

        forbidden_names = (
            "person_nvdcf_tracker",
            "track_osd",
            "native_yolo26_pgie",
            "native_nvdcf_tracker",
        )
        present = [name for name in forbidden_names if self.pipeline.get_by_name(name) is not None]
        if present:
            raise RuntimeError(
                "CAMERA_CLEAN_PIPELINE_AUDIT analytics unexpectedly present: "
                + ", ".join(present)
            )

        if self.frame_width != 1280 or self.frame_height != 720:
            raise RuntimeError(
                f"CAMERA_CLEAN_PIPELINE_AUDIT expected mux=1280x720 got={self.frame_width}x{self.frame_height}"
            )
        if self.wall_width != 1920 or self.wall_height != 720:
            raise RuntimeError(
                f"CAMERA_CLEAN_PIPELINE_AUDIT expected wall=1920x720 got={self.wall_width}x{self.wall_height}"
            )

        print(
            "CAMERA_CLEAN_PIPELINE_AUDIT status=OK "
            "chain=nvurisrcbin/NVDEC->queue1->nvstreammux->nvmultistreamtiler->wall_caps->queue1->nveglglessink "
            "analytics=0",
            flush=True,
        )

    def _startup_stagger_seconds(self) -> float:
        configured = float(getattr(self.settings.deepstream, "startup_stagger_sec", 0.5))
        return max(
            0.10,
            min(
                3.0,
                float(
                    os.environ.get(
                        "CAMERA_V2_STARTUP_STAGGER_SEC",
                        str(configured),
                    )
                ),
            ),
        )

    def _prepare_staggered_sources(self) -> None:
        ordered = [camera.camera_id for camera in self.cameras]
        stagger = self._startup_stagger_seconds()

        for cid in ordered:
            source = self.sources[cid]
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)

        print(
            f"CAMERA_CLEAN_SOURCE_STAGGER order={ordered} interval={stagger:.2f}s",
            flush=True,
        )

        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * stagger * 1000.0)))

            def _start(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                sync = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._last_progress[camera_id] = now
                self._last_frames[camera_id] = int(self.stats[camera_id].frames)
                print(
                    f"CAMERA_CLEAN_SOURCE_START cid={camera_id} index={ordinal} sync={int(sync)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _source_watchdog(self) -> bool:
        if self._stopping:
            return False

        now = time.monotonic()
        for cid, started_at in list(self._source_started_at.items()):
            current = int(self.stats[cid].frames)
            if current != self._last_frames[cid]:
                self._last_frames[cid] = current
                self._last_progress[cid] = now
                continue

            if now - started_at < self._stall_s:
                continue
            stalled = now - self._last_progress[cid]
            if stalled < self._stall_s:
                continue

            self._restart_requested = True
            self._restart_reason = f"{cid} no-frames {stalled:.1f}s"
            print(
                f"CAMERA_CLEAN_PROCESS_RESTART reason={self._restart_reason} "
                f"exit_code={RESTART_EXIT_CODE}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def run(self) -> int:
        self._prepare_staggered_sources()
        self.GLib.timeout_add_seconds(1, self._source_watchdog)
        result = super().run()
        if self._restart_requested:
            return RESTART_EXIT_CODE
        return result


def main() -> int:
    return CleanCameraWallV2().run()


if __name__ == "__main__":
    raise SystemExit(main())

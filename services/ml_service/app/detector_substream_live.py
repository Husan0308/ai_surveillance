from __future__ import annotations

import signal
import time

from services.ml_service.app.detector_substream import DetectorSubstreamService


class DetectorSubstreamLiveService(DetectorSubstreamService):
    """Live-source lifecycle fix for the dual-stream detector.

    The base detector intentionally keeps every RTSP source locked in NULL while
    the parent pipeline transitions, then releases sources one at a time to avoid
    a six-camera connection storm. A GstBaseSink normally waits for preroll while
    entering PAUSED. That is incompatible with locked live RTSP sources because a
    live source only produces buffers in PLAYING.

    appsink is therefore made non-async for this real-time capture graph. The
    parent can reach PLAYING without waiting for a preroll frame; each RTSP source
    is then unlocked and synchronised directly to the parent's PLAYING state.
    """

    @staticmethod
    def _state_name(value) -> str:
        return str(getattr(value, "value_nick", value))

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        sink = self.pipeline.get_by_name(f"ml_sub_sink_{index}")
        if sink is None:
            raise RuntimeError(f"{camera.camera_id}: appsink missing after graph build")

        # GstBaseSink async=FALSE removes the preroll barrier. sync=FALSE was
        # already configured by the base runtime; the remaining properties keep
        # this sink latest-only and outside any clock/QoS feedback loop.
        self._set_if(sink, "async", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "processing-deadline", 0)
        self._set_if(sink, "max-lateness", -1)

    def _start_sources(self) -> None:
        for source in self.sources.values():
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)

        transition = self.pipeline.set_state(self.Gst.State.PLAYING)
        state_ret, current, pending = self.pipeline.get_state(0)
        print(
            "ML_SUBSTREAM_STATE "
            f"pipeline_target=PLAYING set_state={self._state_name(transition)} "
            f"query={self._state_name(state_ret)} current={self._state_name(current)} "
            f"pending={self._state_name(pending)} sink_async=0",
            flush=True,
        )

        for camera in self.cameras:
            source = self.sources[camera.camera_id]
            source.set_locked_state(False)
            synced = bool(source.sync_state_with_parent())
            state_ret, current, pending = source.get_state(0)
            print(
                "ML_SUBSTREAM_STATE "
                f"{camera.camera_id} sync_parent={int(synced)} "
                f"query={self._state_name(state_ret)} current={self._state_name(current)} "
                f"pending={self._state_name(pending)}",
                flush=True,
            )
            time.sleep(self.startup_stagger_sec)


def main() -> int:
    service = DetectorSubstreamLiveService()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())

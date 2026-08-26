from __future__ import annotations

import signal

from services.ml_service.app.detector_substream_pts_burst import DetectorSubstreamPtsBurstService


class DetectorSubstreamRtpPtsService(DetectorSubstreamPtsBurstService):
    """V10 PTS-burst detector with receive-time TCP timestamping disabled.

    GStreamer's rtspsrc documentation notes that TCP servers commonly burst data.
    With tcp-timestamp=true every RTP packet is timestamped from receive time, which
    can make decoded PTS follow burst/stall arrival timing instead of the smoother
    RTP media timeline. V10 gates sparse preprocessing by decoded PTS, so receive-
    time timestamping defeats that scheduler on CAM-02.

    Keep TCP transport and every V10 detector/pending-queue setting unchanged, but
    leave tcp-timestamp disabled so rtspsrc interpolates timing from RTP timestamps.
    This is intentionally ML-substream-only; Camera Service is untouched.
    """

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(_bin, _sub_bin, element, camera)
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if self.rtsp_transport == "tcp":
            # rtspsrc defaults this to false. Set it explicitly because the base
            # detector used true for an earlier receive-time experiment.
            self._set_if(element, "tcp-timestamp", False)
            print(
                f"ML_SUBSTREAM_RTP_CLOCK {camera.camera_id} transport=tcp "
                "tcp_timestamp=0 clock=rtp-interpolated",
                flush=True,
            )


def main() -> int:
    service = DetectorSubstreamRtpPtsService()

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

from __future__ import annotations

from .step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7


class V11Step1BurstBackpressureV20(V11Step1Cam02LowLatV7):
    """Depth-1 non-leaky display queue for short decoded-frame bursts."""

    def _configure_latest_queue(self, queue) -> None:
        super()._configure_latest_queue(queue)
        self._set_if(queue, "leaky", 0)

    def __init__(self) -> None:
        super().__init__()
        effective = {
            cid: int(queue.get_property("leaky")) for cid, queue in self.queues.items()
        }
        if any(value != 0 for value in effective.values()):
            raise RuntimeError(f"V20 display queue policy not effective: {effective}")
        print(
            "CAMERA_V11_STEP1_V20_QUEUE policy=bounded-backpressure max_buffers=1 "
            "leaky=0 growing_backlog=0 quality_changed=0 effective="
            + ",".join(f"{cid}:{value}" for cid, value in sorted(effective.items())),
            flush=True,
        )


def main() -> int:
    return V11Step1BurstBackpressureV20().run()


if __name__ == "__main__":
    raise SystemExit(main())

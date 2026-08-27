#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

RTP = re.compile(
    r"CAMERA_V109_RTP camera=(CAM-\d+) index=(\d+) pushed=(\d+) pushed_5s=(\d+) "
    r"lost=(\d+) loss_pct=([0-9.]+) late=(\d+) late_pct=([0-9.]+) "
    r"duplicates=(\d+) avg_jitter_ms=([0-9.]+)"
)
SRC = re.compile(
    r"CAMERA_V105_SOURCE camera=(CAM-\d+) input_count=(\d+) input_vs_max=([0-9.]+)% "
    r"arrival_p50=([0-9.]+)ms arrival_p95=([0-9.]+)ms arrival_p99=([0-9.]+)ms "
    r"mux_latest_lag_p50=([0-9.]+)ms mux_latest_lag_p95=([0-9.]+)ms"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v109_log.py /tmp/CAMERA_BBOX_V109.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V109_ARCH" not in text:
        print("V109_RTP_AUDIT FAIL missing=CAMERA_V109_ARCH")
        return 2

    rtp_last: dict[str, tuple[float, float, float, int]] = {}
    for m in RTP.finditer(text):
        cid = m.group(1)
        # Prefer the video stream with the largest cumulative pushed count if
        # more than one RTP stream is present for a camera.
        row = (float(m.group(10)), float(m.group(6)), float(m.group(8)), int(m.group(3)))
        old = rtp_last.get(cid)
        if old is None or row[3] >= old[3]:
            rtp_last[cid] = row

    if "CAM-02" not in rtp_last or len(rtp_last) < 4:
        missing = [cid for cid in [f"CAM-0{i}" for i in range(1, 7)] if cid not in rtp_last]
        print(f"V109_RTP_AUDIT FAIL rtp_stats_unavailable=1 cameras={sorted(rtp_last)} missing={missing}")
        print("V109_RTP_AUDIT next=send CAMERA_V109_MANAGER/JB_READY/RTP lines; do not tune mux or NvDCF")
        return 2

    src_last: dict[str, float] = {}
    for m in SRC.finditer(text):
        src_last[m.group(1)] = float(m.group(5))

    cam_jitter, cam_loss, cam_late, cam_pushed = rtp_last["CAM-02"]
    peers = [row for cid, row in rtp_last.items() if cid != "CAM-02"]
    peer_jitter = statistics.median(row[0] for row in peers)
    peer_loss = statistics.median(row[1] for row in peers)
    peer_late = statistics.median(row[2] for row in peers)
    peer_arrival = statistics.median(
        value for cid, value in src_last.items() if cid != "CAM-02"
    ) if len(src_last) >= 4 else 0.0
    cam_arrival = src_last.get("CAM-02", 0.0)

    jitter_outlier = cam_jitter >= max(peer_jitter * 1.40, peer_jitter + 2.0)
    loss_outlier = cam_loss >= max(peer_loss * 2.0 + 0.02, 0.05)
    late_outlier = cam_late >= max(peer_late * 2.0 + 0.02, 0.05)
    arrival_outlier = bool(cam_arrival and peer_arrival and cam_arrival >= peer_arrival + 25.0)

    if jitter_outlier or loss_outlier or late_outlier:
        diagnosis = "RTP_OR_NVR_PACKET_JITTER"
        next_step = "inspect CAM-02 NVR/camera stream packet cadence, loss and encoder settings; keep TCP60 baseline"
    elif arrival_outlier:
        diagnosis = "RTP_HEALTHY_BUT_DECODED_CADENCE_OUTLIER"
        next_step = "audit CAM-02 encoded access-unit/decode cadence before tracker mux; do not tune RTSP transport again"
    else:
        diagnosis = "RTP_AND_SOURCE_CADENCE_COMPARABLE"
        next_step = "return to tracker stage audit; CAM-02 is no longer a source-side outlier"

    print(
        "V109_RTP_AUDIT RESULT "
        f"diagnosis={diagnosis} cam02_jitter_ms={cam_jitter:.3f} peer_jitter_ms={peer_jitter:.3f} "
        f"cam02_loss_pct={cam_loss:.4f} peer_loss_pct={peer_loss:.4f} "
        f"cam02_late_pct={cam_late:.4f} peer_late_pct={peer_late:.4f} "
        f"cam02_arrival_p95={cam_arrival:.0f}ms peer_arrival_p95={peer_arrival:.0f}ms pushed={cam_pushed}"
    )
    print(f"V109_RTP_AUDIT next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

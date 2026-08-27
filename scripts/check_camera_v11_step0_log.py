#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

PAT = re.compile(
    r"CAMERA_V11_STEP0_STATS camera=(\S+) decoded_total=(\d+) decode_fps=([0-9.]+) sink_fps=([0-9.]+) "
    r"wall_p50=([0-9.]+)ms wall_p95=([0-9.]+)ms wall_p99=([0-9.]+)ms "
    r"pts_p50=([0-9.]+)ms pts_p95=([0-9.]+)ms q=(\d+) qmax=(\d+) "
    r"rtp_pushed=(\d+) rtp_lost=(\d+) rtp_late=(\d+) rtp_dup=(\d+) "
    r"rtp_jitter_ms=([0-9.]+) errors=(\d+) warnings=(\d+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step0_log.py /tmp/CAMERA_V11_STEP0.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP0_ARCH",
        "CAMERA_V11_STEP0_POLICY",
        "CAMERA_V11_STEP0_INVARIANT",
    ):
        if marker not in text:
            print(f"V11_STEP0 FAIL missing={marker}")
            return 2

    latest: dict[str, re.Match[str]] = {}
    for match in PAT.finditer(text):
        latest[match.group(1)] = match
    if len(latest) < 6:
        print(f"V11_STEP0 FAIL cameras={len(latest)} expected=6")
        return 2

    rows = []
    for cid, m in sorted(latest.items()):
        rows.append(
            {
                "cid": cid,
                "decoded": int(m.group(2)),
                "fps": float(m.group(3)),
                "sink_fps": float(m.group(4)),
                "wall95": float(m.group(6)),
                "wall99": float(m.group(7)),
                "pts95": float(m.group(9)),
                "qmax": int(m.group(11)),
                "pushed": int(m.group(12)),
                "lost": int(m.group(13)),
                "late": int(m.group(14)),
                "jitter": float(m.group(16)),
                "errors": int(m.group(17)),
            }
        )

    max_fps = max(row["fps"] for row in rows)
    peer_jitters = [row["jitter"] for row in rows if row["pushed"] > 100]
    median_jitter = statistics.median(peer_jitters) if peer_jitters else 0.0
    fatal = []
    jitter_outliers = []

    for row in rows:
        ratio = 100.0 * row["fps"] / max(0.001, max_fps)
        delivery = 100.0 * row["sink_fps"] / max(0.001, row["fps"])
        loss_pct = 100.0 * row["lost"] / max(1, row["pushed"] + row["lost"])
        late_pct = 100.0 * row["late"] / max(1, row["pushed"] + row["late"])
        if row["decoded"] < 200:
            fatal.append(f"{row['cid']}:insufficient_frames")
        if ratio < 85.0:
            fatal.append(f"{row['cid']}:fps_ratio={ratio:.1f}%")
        if delivery < 90.0:
            fatal.append(f"{row['cid']}:delivery={delivery:.1f}%")
        if row["wall95"] > 220.0:
            fatal.append(f"{row['cid']}:arrival_p95={row['wall95']:.0f}ms")
        if row["qmax"] > 1:
            fatal.append(f"{row['cid']}:qmax={row['qmax']}")
        if row["errors"] > 0:
            fatal.append(f"{row['cid']}:errors={row['errors']}")
        if loss_pct > 0.05 or late_pct > 0.05:
            fatal.append(f"{row['cid']}:loss={loss_pct:.3f}%/late={late_pct:.3f}%")
        if row["pushed"] > 100 and row["jitter"] > max(5.0, median_jitter * 2.5):
            jitter_outliers.append(row["cid"])

        print(
            "V11_STEP0_CAMERA "
            f"camera={row['cid']} fps={row['fps']:.2f} ratio={ratio:.1f}% "
            f"delivery={delivery:.1f}% wall_p95={row['wall95']:.0f}ms "
            f"pts_p95={row['pts95']:.0f}ms qmax={row['qmax']} "
            f"jitter={row['jitter']:.3f}ms loss={loss_pct:.4f}% late={late_pct:.4f}%"
        )

    if fatal:
        print("V11_STEP0 RESULT diagnosis=FAIL_SOURCE_OR_DECODE reasons=" + ";".join(fatal))
        print("V11_STEP0 next=do not add display; fix only the failing ingest/decode condition")
        return 1

    if jitter_outliers:
        print(
            "V11_STEP0 RESULT diagnosis=PASS_SOURCE_JITTER_ISOLATED "
            f"jitter_outliers={','.join(jitter_outliers)} median_jitter_ms={median_jitter:.3f}"
        )
        print(
            "V11_STEP0 next=freeze Step0; external jitter is measured but isolated, then build Step1 display without changing ingest"
        )
    else:
        print(
            "V11_STEP0 RESULT diagnosis=PASS_CLEAN_INGEST "
            f"median_jitter_ms={median_jitter:.3f}"
        )
        print("V11_STEP0 next=freeze Step0 and add display-only Step1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

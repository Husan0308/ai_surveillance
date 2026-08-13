from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    end = time.monotonic() + max(1, args.seconds)
    scores = {}
    last_metrics = None
    seen = set()

    while time.monotonic() < end:
        payload = fetch(args.base.rstrip("/") + "/reid")
        state = payload.get("state") or {}
        last_metrics = payload.get("metrics") or {}
        for row in state.get("recent_pair_scores") or []:
            key = (row.get("pair"), row.get("left"), row.get("right"), row.get("ts"))
            if key in seen:
                continue
            seen.add(key)
            pair = str(row.get("pair"))
            scores.setdefault(pair, []).append(float(row.get("score", 0.0)))
        time.sleep(max(0.2, args.interval))

    print(json.dumps({
        "algorithm": (last_metrics or {}).get("algorithm"),
        "model": (last_metrics or {}).get("model"),
        "ready": (last_metrics or {}).get("ready"),
        "last_error": (last_metrics or {}).get("last_error"),
        "pair_merges": (last_metrics or {}).get("pair_merges"),
        "pair_confirms": (last_metrics or {}).get("pair_confirms"),
        "pair_rejects": (last_metrics or {}).get("pair_rejects"),
        "descriptor_updates": (last_metrics or {}).get("descriptor_updates"),
        "pairs": {
            pair: {
                "count": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            for pair, values in sorted(scores.items()) if values
        },
    }, indent=2))


if __name__ == "__main__":
    main()

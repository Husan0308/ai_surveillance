#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
EXPECTED_BRANCH="rebuild/clean-production-v1-20260920"
STAGE="${1:-quick}"
STAMP="${SIX_CAMERA_GATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_BASE="${SIX_CAMERA_GATE_OUT:-$ROOT/.runtime/six-camera-freeze-gate-$STAMP}"
LATENCY_MS="${SIX_CAMERA_GATE_LATENCY_MS:-100}"

fail() {
  printf 'SIX_CAMERA_FREEZE_GATE status=FAIL stage=%s reason=%s\n' "$STAGE" "$*" >&2
  exit 1
}

current_branch="$(git branch --show-current 2>/dev/null || true)"
[[ "$current_branch" == "$EXPECTED_BRANCH" ]] || fail "wrong_branch=${current_branch:-detached} expected=$EXPECTED_BRANCH"

[[ "$LATENCY_MS" == "100" ]] || fail "freeze_baseline_requires_latency_ms=100 got=$LATENCY_MS"

mkdir -p "$OUT_BASE"

run_unit() {
  echo "=== SIX-CAMERA UNIT GATE ==="
  python3 -m unittest tests.test_cam_six_stability -v
}

run_quick() {
  local out="$OUT_BASE/quick"
  echo "=== SIX-CAMERA QUICK GATE (60 s, camera-only, no AI) ==="
  rm -rf "$out"
  python3 scripts/validate_cam_six.py     --duration 60     --latency-ms "$LATENCY_MS"     --out "$out"     --no-preview
  echo "SIX_CAMERA_QUICK status=PASS out=$out"
}

run_soak() {
  local out="$OUT_BASE/soak"
  echo "=== SIX-CAMERA SOAK GATE (660 s, camera-only, no AI) ==="
  rm -rf "$out"
  python3 scripts/validate_cam_six.py     --duration 660     --latency-ms "$LATENCY_MS"     --out "$out"     --no-preview
  python3 scripts/cam_six_validation/check_stability.py "$out"
  python3 - "$out/stability.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
if data.get("status") != "PASS":
    raise SystemExit("six-camera soak report is not PASS")
for cid in ("CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06"):
    row = data.get(cid) or {}
    mean = row.get("fps_mean")
    if mean is None or not 18 <= float(mean) <= 22:
        raise SystemExit(f"{cid} invalid fps_mean={mean}")
print(json.dumps({
    "status": data["status"],
    "cameras": {
        cid: {
            "fps_mean": data[cid]["fps_mean"],
            "max_gap_ms": data[cid]["max_gap_ms"],
            "queue_max": data[cid]["queue_max"],
        }
        for cid in ("CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06")
    }
}, indent=2))
PY
  echo "SIX_CAMERA_SOAK status=PASS out=$out"
}

run_isolation() {
  echo "=== SIX-CAMERA ISOLATION/RECOVERY GATE ==="
  for cid in CAM-01 CAM-02 CAM-03 CAM-04 CAM-05 CAM-06; do
    local out="$OUT_BASE/isolation-$cid"
    rm -rf "$out"
    echo "--- interrupting $cid while five peers must stay healthy ---"
    python3 scripts/validate_cam_six.py       --duration 60       --latency-ms "$LATENCY_MS"       --interrupt-camera "$cid"       --interrupt-at 20       --interrupt-seconds 12       --out "$out"       --no-preview
    echo "SIX_CAMERA_ISOLATION camera=$cid status=PASS out=$out"
  done
  echo "SIX_CAMERA_ISOLATION status=PASS"
}

case "$STAGE" in
  quick)
    run_unit
    run_quick
    ;;
  soak)
    run_unit
    run_soak
    ;;
  isolation)
    run_unit
    run_isolation
    ;;
  all)
    run_unit
    run_quick
    run_soak
    run_isolation
    ;;
  *)
    fail "usage: $0 {quick|soak|isolation|all}"
    ;;
esac

cat >"$OUT_BASE/gate-summary.json" <<EOF
{
  "status": "PASS",
  "stage": "$STAGE",
  "branch": "$EXPECTED_BRANCH",
  "latency_ms": $LATENCY_MS,
  "scope": ["CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06"],
  "analytics_changed": false,
  "output_root": "$OUT_BASE"
}
EOF

echo "SIX_CAMERA_FREEZE_GATE status=PASS stage=$STAGE out=$OUT_BASE"

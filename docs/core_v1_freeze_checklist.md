# Core v1 freeze checklist

Core v1 is designed as a latest-only realtime baseline. No identity/ReID feature may rely on predicted visual boxes.

## Run sequence

1. Start ML core and frontend.
2. Perform a 30-minute burn-in:
   `python scripts/core_v1_soak.py --minutes 30 --output core_v1_burnin.jsonl`
3. Perform human detection checks on all six cameras.
4. Perform a 3-hour soak:
   `python scripts/core_v1_soak.py --minutes 180 --interval 10 --output core_v1_3h.jsonl`
5. Freeze only after the acceptance conditions below pass.

## Realtime acceptance

- Camera post-decode queue stays bounded at 0-1 buffers.
- `last_frame_age_ms` remains low and does not trend upward over time.
- `pipeline_lag_ms` does not accumulate continuously. A single-camera drift reconnect is acceptable if it recovers promptly; repeated reconnect loops are not.
- Detector `queue_full_drops` should remain zero in normal operation.
- `finish_age_ms.p95` should remain below the configured `max_result_age_ms`; stale drops may occur under load, but their rate must not rise continuously.
- Publisher `last_publish_source_age_ms` should not show an increasing trend.
- Detector batch p95 and max should remain stable between the first and last 30 minutes of the 3-hour soak.

## Resource acceptance

Compare the first stable 10 minutes with the last stable 10 minutes. No strict absolute RSS/VRAM value is assumed because model/runtime versions differ.

- Detector RSS must not show monotonic unbounded growth.
- Detector GPU process memory must plateau after warm-up and must not show monotonic unbounded growth.
- Thread count and file-descriptor count must remain approximately stable after startup/reconnect activity settles.
- GPU temperature/utilization may vary with scene activity, but thermal throttling must not cause a sustained detector cadence collapse.

## Detection acceptance

- CAM-06 physical TV never creates a visible person track during the test sequence.
- A real standing person in front of/near the TV remains detectable.
- A person lying on the CAM-06 sofa is recovered by the conditional native ROI without rotated ROI passes.
- Static chair/desk clutter does not create persistent person tracks.
- One physical person does not produce persistent duplicate boxes.
- Two close real people remain two tracks; fragment suppression must not merge them.
- Moving boxes do not visibly trail for long periods. Old detector observations are rejected rather than painted over newer frames.

## Architecture invariants

Do not regress these rules when adding later features:

- one `LatestFrameStore` slot per camera;
- decoded-frame dropping only after decoder, never compressed H264/H265 packet dropping before decoder;
- one detector batch in flight;
- stale frame rejection before inference;
- stale detector result rejection after inference/publish;
- detector-level static hard masks before tracker birth;
- ROI second pass is selective/conditional and bounded;
- strong new track = repeated evidence, medium new track = more repeated evidence, low score = existing-track continuation only;
- long identity memory and short visible-box lifetime are separate concepts;
- visual prediction is never identity/ReID evidence.

## Rollback points

If a regression appears, revert the smallest subsystem first:

1. ROI config only.
2. Camera-specific hard-mask/birth-zone config only.
3. Visual tracker confirmation/prediction settings.
4. Detector freshness/telemetry patch.
5. GStreamer capture/watchdog patch.

Do not compensate for a latency regression by adding another queue or increasing visible hold time.

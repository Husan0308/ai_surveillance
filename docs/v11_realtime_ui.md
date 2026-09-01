# V11 realtime camera UI

Branch: `rebuild/service-architecture-v11-ui-realtime-cameras-v1-20260901`

This milestone wires the desktop camera wall to the service architecture without enabling ReID/global identity in the UI path.

## Data path

```text
RTSP cameras
  -> ml_service DeepStream video publisher
  -> API MJPEG proxy
  -> Qt MjpegStream latest-only decoder
  -> CameraTile

V11 detector + local tracker metadata
  -> ml_service /tracks
  -> api_service /api/v1/ws/monitoring
  -> MonitoringSocket
  -> LatestMetadataStore
  -> CameraTile bbox overlay
```

The UI talks only to `api_service`; it does not connect directly to the ML service.

## Run

From the repository root, use three terminals:

```bash
python3 -m services.ml_service.app.main
```

```bash
python3 -m services.api_service.app.main
```

```bash
python3 -m services.frontend.app.main
```

Defaults:

- API: `http://127.0.0.1:8000`
- ML: `http://127.0.0.1:8001`
- UI REST base: `FRONTEND_API_BASE_URL=http://127.0.0.1:8000`
- UI REST refresh: `FRONTEND_REFRESH_INTERVAL_MS=2000`
- UI JPEG decode period: `FRONTEND_FRAME_REFRESH_INTERVAL_MS=33`
- UI WebSocket reconnect: `FRONTEND_WS_RECONNECT_MS=1000`
- stale tracker overlay cutoff: `ML_V11_MONITOR_STALE_SEC=1.50`

## UI contract

For each camera, the UI receives the MJPEG frame and tracker metadata independently. The bbox mapper uses the metadata source dimensions and the actual widget letterbox rectangle, so the overlay remains aligned when a tile is resized.

Tracked boxes are green. Predicted/coasting/lost boxes are yellow. Old metadata is rejected by timestamp/frame sequence, and ML-side stale metadata is cleared after the configured cutoff instead of leaving a frozen bbox on live video.

## Network connection pools

There are two `QNetworkAccessManager` instances by design:

- control manager: health/camera REST requests;
- stream manager: the six long-lived MJPEG streams.

Qt HTTP/1 executes six requests in parallel per host/port for a manager. Keeping six persistent streams in their own manager prevents them from occupying the whole control-plane pool.

## Tests

Pure realtime model/geometry tests do not require a running camera stack:

```bash
pytest -q tests/test_frontend_realtime_models.py
```

Then perform the hardware acceptance with all six cameras:

1. `/health` on API is `ok`.
2. `/api/v1/ml/health` reports the expected online camera count and `tracking.ready=true`.
3. `/api/v1/cameras` returns six real cameras and API stream URLs.
4. All six tiles show live video without scroll-dependent hidden feeds.
5. A visible person receives a `PERSON · T...` bbox on the correct camera.
6. When tracker metadata stops, the old bbox disappears rather than freezing indefinitely.
7. REST health/camera refresh continues while all six MJPEG streams are open.

This milestone deliberately does not add face recognition, ReID, global ID, pose, or another inference pass.

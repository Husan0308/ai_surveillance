# Surveillance application runtime (F3)

AutoMagicCalib owns port 8000 and is not moved or managed by this application.
The surveillance contract is centralized in `services/shared/deployment.py`.

| Variable | Default |
| --- | --- |
| API_HOST / API_PORT | 0.0.0.0 / 8100 |
| ML_HOST / ML_PORT | 0.0.0.0 / 8101 |
| ML_SERVICE_URL | http://127.0.0.1:8101 |
| FRONTEND_API_BASE_URL | http://127.0.0.1:8100 |
| FRONTEND_ML_VIDEO_BASE_URL | http://127.0.0.1:8101 |

Port overrides derive corresponding URLs unless explicitly overridden.
Monitoring websocket: API base URL + `/ws/v1/monitoring`, ws/wss scheme.
The ML video URL is compatibility metadata, **not** an MJPEG transport.

Run `bash scripts/start_full_live_stack.sh --preflight` for the frozen F1 import
check; run without arguments to start the supervised camera-only application.
It uses `.runtime/full-stack-venv` and the frozen F2 DeepStream 9.1 camera owner.
No detector/tracker acceptance is implied by this F3 launcher.
The UI never opens RTSP or MJPEG. All camera rows use `camera_id`.

The supervisor fails closed on occupied ports, source locks, missing credentials,
image, or display. `FULL_STACK_OUT` selects a **new** evidence directory.
Closing the wall or interrupting the supervisor shuts down owned children.
F1 Python resolution is unchanged; old orchestration below preflight is
intentionally replaced by F3 supervision.

A latest-only telemetry bridge reads preview headers and producer statistics;
it adds no work to decode/preview. A file alone is not live evidence: timestamps
must belong to the current session, be fresh, and advance. Stopped producers
become offline even if final stats say LIVE. Missing counters remain null.
ML requires six current producers; API degrades when ML disappears.
Camera-only mode explicitly disables room analytics and exposes no replay state.

Acceptance: `http://127.0.0.1:8101/health`,
`http://127.0.0.1:8100/health`, `/api/v1/cameras`,
`/api/v1/monitoring/snapshot`, `/ws/v1/monitoring` on the API.
`FULL_STACK_UI_EVIDENCE=1` uses the same production MainWindow on the actual
display and records tile sequences/status/fullscreen evidence. It does not
substitute offscreen/replay video for live acceptance.

Run camera service SHM:
  bash scripts/run_camera_service_shm.sh 2>&1 | tee /tmp/CAMERA_SERVICE_SHM.log

In another terminal:
  python scripts/probe_camera_service_shm.py --seconds 30

Acceptance:
  camera display remains ~20 FPS for all six cameras
  SHM probe reports ~2 Hz for all six cameras
  age_max remains below 500 ms

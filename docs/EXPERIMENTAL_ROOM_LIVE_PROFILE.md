# Experimental room detector/tracker staging

The accepted PeopleNet profile, native binary, calibration and asset manifest are
not replaced. The bare production validator still reports the pre-existing
source/identity/binary drift. This tooling does not suppress that result.

`scripts/build_room_candidate.py` builds a separate native executable from the
current checkout and pinned DeepStream 9.1 toolchain. It records every native/SDK
compile-input hash, staged source hash, ELF build ID and binary hash. Launch
refuses changed inputs or binary bytes. Two explicitly staged C diagnostics are
used: skip tracker-specific bbox probes only when tracking is disabled; include
unchanged object rectangles in the existing environment-gated frame audit.
Tracker-enabled execution retains the existing bbox correction call.

`scripts/run_room_candidate.py` creates a new private profile under `.runtime`.
The committed YOLO26m raw one-to-many detector config is copied, changing only
requested batch size 6→2; the preserved FP16 engine supports batches 1–6.
Confidence 0.25, NMS 0.45, 640×640 RGB letterbox and interval 0 remain unchanged.
The app-level engine override is also changed explicitly so PeopleNet cannot
silently win over the detector config. Engine, ONNX and parser mounts are
read-only. The engine's recorded samples image is used (TensorRT 10.16.1.11);
the older 9.1 Triton image's TensorRT 10.16.0.72 cannot deserialize this engine.

The project-local ignored model-cache symlink points to the preserved validated
YOLO cache. Launch verifies the cache hashes, current GPU and platform contract,
not merely file existence. No model is downloaded or rebuilt on launch.

F4 detector-only diagnostics disable NvDCF and the Kafka identity sink. Native
readiness therefore correctly remains WARMING: F4 is not identity acceptance.
Both RTSP ownership locks are held until native shutdown. Existing independent
CUDA preview, source settings, queue limits and mux settings remain unchanged.
The frontend never opens RTSP.

Evidence includes the unsuccessful runtime-mismatch smoke, exact loaded config
hashes, native frame audit, decoder correlation, queue levels, actual sockets,
GPU sampling, and live detector overlays. Overlay metadata records the PTS delta
between each independent-preview image and its nearest PGIE frame; it is not a
claim of manually labelled person recall. A clean owned SIGINT is retained in
logs even though the sample app labels `User Interrupted` at ERROR severity.
Other error messages are never excused.

F4 validation is an experimental live detector freeze, not production promotion.
Production live person-recall acceptance and absolute metric calibration remain
separate requirements.

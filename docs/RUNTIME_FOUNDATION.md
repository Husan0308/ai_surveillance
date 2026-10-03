# Host Python runtime (F1)

API, ML, PySide6, preview-only workers, room-pair orchestration, Kafka capture and
the OSNet identity worker use **one default interpreter**:
`.runtime/full-stack-venv/bin/python`. Source code stays in the checkout.

Ubuntu 24.04 `/usr/bin/python3` (3.12) creates this environment with
`--system-site-packages` for Ubuntu's ABI-matched `python3-gi` / GStreamer 1.0.
User-site and ambient `PYTHONPATH`/`PYTHONHOME` are not runtime dependencies.
Application packages, including OSNet's Python dependencies and test tools, are
pinned in `requirements/runtime-host.lock.txt` and installed locally. The service
requirements remain interface-level ranges; this lock is the deployment closure.

## NVIDIA boundary

Native analytics runs in the existing pinned DeepStream 9.1 image, with its
CUDA 13.2 / TensorRT 10.16.x toolchain. This host setup does not install,
upgrade, downgrade or inject CUDA, TensorRT, NVIDIA bindings, GI, or drivers into
that image or the host system. It does not use pyservicemaker.

OSNet still requires GPU PyTorch. Rather than installing another CUDA package
stack, setup requires an **explicit existing non-Trash GPU library provider**.
It creates package-specific symlinks for PyTorch/torchvision, their auxiliary
modules, Triton and CUDA/NVIDIA namespaces and distribution metadata. It does
not add the provider's whole site-packages directory to `sys.path`, use its
interpreter, or import its API/UI packages. No provider files are copied or
changed. These shared platform libraries must be retained, not pip-upgraded in
place. `.runtime/full-stack-venv/platform-libs.json` declares exact link targets
and hashes of package metadata/RECORDs/module initializers; preflight rejects
drift, missing links and undeclared import origins. These are existing CUDA 12.6
PyTorch runtime libraries, **not** the DeepStream container toolchain.

The currently inventoried provider is the existing sibling checkout's
`/home/apsidal/ai_surveillance/.venv/lib/python3.12/site-packages`, containing
`torch==2.13.0+cu126`, `torchvision==0.28.0+cu126`, `triton==3.7.1`.
This is a declared, pinned platform dependency, not a hidden global fallback.
Other providers must satisfy those exact versions. Model files/OSNet logic and
identity thresholds are unchanged; live identity acceptance is a later freeze.

## Explicit setup and camera-free validation

```bash
bash scripts/setup_runtime.sh \
  --gpu-site /home/apsidal/ai_surveillance/.venv/lib/python3.12/site-packages
bash scripts/start_full_live_stack.sh --preflight
PYTHONNOUSERSITE=1 .runtime/full-stack-venv/bin/python -m pytest tests -q
```

First setup creates/reuses only the one project venv under a setup lock. It
installs the full pinned non-NVIDIA closure with `--no-deps` (no unconstrained
NVIDIA resolution). Subsequent setup calls perform no installation when
preflight already passes. An incompatible existing runtime is not destroyed or
silently replaced. The launcher never auto-installs or auto-creates an environment.
Missing modules cause a nonzero exit with the failed import/version on stderr.
`--preflight` exits before output directories, locks, ports, processes, or cameras.

The launcher helper/setup stdout is only an interpreter path. Diagnostics go to
stderr; launcher preflight stdout is structured JSON. `Gst.init` discovers
plugins only; it does not create pipelines. Kafka checks only import and
round-trip the repository protobuf schema, never connect a consumer. CUDA
availability is checked for OSNet without opening cameras.

Explicit interpreter overrides (absolute paths, same preflight requirements):

- `FULL_STACK_PYTHON`: launcher and default for all room-pair Python workers.
- `MV3DT_OSNET_PYTHON`: identity worker override, higher priority than the above.
- `MV3DT_KAFKA_PYTHON`: Kafka capture override, higher priority than the above.

No implicit `VIRTUAL_ENV`, PATH Python, old venv or Trash fallback is permitted.
The old `setup_v91_py310_runtime_deps.sh` name is only a deprecated forwarding
alias; `CAMERA_V91_PYTHON` is retired. Use the commands above instead.

## Scope / historical material

`docs/ARCHITECTURE_V2.md` describes a retired TRT86 branch, not this runtime.
The migration audit's old paths are historical evidence. Legacy `config/reid.yaml`
is retained/protected future-layer configuration, not selected by the host
runtime. F1 does not modify those model/config assets. Absolute NVIDIA reference
app/artifact paths in the Dev Room profile remain platform artifact locations,
not interpreter selectors. Port ownership, live process ownership and UI status
remain later freezes; a passing Python preflight is not camera/production-asset
acceptance. Known F0 asset-manifest drift is not rebaselined here.

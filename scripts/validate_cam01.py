#!/usr/bin/env python3
"""Run only the existing CAM-01 in the pinned DS9.1 image; never run AI/other cameras."""
from pathlib import Path
import argparse
import base64
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.shared.camera_config import load_settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['decode', 'display', 'record'], default='record')
    ap.add_argument('--duration', type=int, default=660)
    ap.add_argument('--latency-ms', type=int, default=300)
    ap.add_argument('--out', type=Path, default=ROOT / '.runtime/cam01-validation')
    args = ap.parse_args()
    if args.duration < 5 or args.latency_ms < 100:
        ap.error('duration >=5 and latency >=100 ms required for this stability profile')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = open('/tmp/ai_surveillance_camera_v2_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = Path(os.getenv('CAMERA_CONFIG', str(ROOT / 'config/cameras.yaml')))
    camera = next(c for c in load_settings(cfg).cameras if c.camera_id == 'CAM-01')
    platform = json.loads((ROOT / 'config/deepstream-platform.json').read_text())
    # Require the previously measured platform gate, rather than rerun a broad preflight per connection.
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'], text=True)
    from scripts.preflight_deepstream91 import version
    name, driver = [s.strip() for s in gpu.splitlines()[0].split(',')]
    if name != platform['gpu_name'] or version(driver) < version(platform['minimum_driver']):
        raise RuntimeError('CAM-01 GPU/driver platform gate failed')
    image = platform['image']
    base = ['docker', 'run', '--rm', '--pull=never', '--user', f'{os.getuid()}:{os.getgid()}']
    build = base + ['--network=none', '-v', f'{ROOT / "scripts/cam01_validation"}:/src:ro',
                    '-v', f'{out}:/out', '--entrypoint', 'bash', image, '-c',
                    'g++ -O2 -std=c++17 -Wall -Wextra /src/main.cpp -o /out/cam01-validator $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0) -lX11']
    subprocess.run(build, check=True)
    secret = out / 'camera-secret.ini'
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('[camera]\n')
        for key, value in [('uri', camera.uri), ('username', camera.username), ('password', camera.password)]:
            f.write(key + '=' + base64.b64encode(value.encode()).decode() + '\n')
        f.write(f'latency_ms={args.latency_ms}\n')
    container = 'ai-surveillance-cam01-validation'
    cmd = base + ['--name', container, '--hostname', socket.gethostname(), '--gpus', 'device=0',
                  '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display',
                  '-e', 'GST_DEBUG=1', '-e', 'GST_REGISTRY=/tmp/cam01-gst-registry.bin',
                  '-v', f'{out}:/work', '--entrypoint', '/work/cam01-validator']
    if args.mode == 'display':
        auth = os.environ.get('XAUTHORITY', '')
        if not auth or not Path(auth).is_file():
            secret.unlink(); raise RuntimeError('XAUTHORITY unavailable; do not alter the video path to fix display')
        cmd += ['-e', 'DISPLAY=' + os.environ.get('DISPLAY', ':0'), '-e', 'XAUTHORITY=/tmp/cam01.xauth',
                '-v', f'{auth}:/tmp/cam01.xauth:ro', '-v', '/tmp/.X11-unix:/tmp/.X11-unix:ro']
    cmd += [image, '/work/camera-secret.ini', str(args.duration), args.mode]
    done = threading.Event()
    def monitor():
        with (out / 'gpu.csv').open('w') as f:
            f.write('time,name,driver,memory_used_mib,gpu_pct,decoder_pct,encoder_pct\n')
            while not done.is_set():
                r = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.used,utilization.gpu,utilization.decoder,utilization.encoder','--format=csv,noheader,nounits'], capture_output=True, text=True)
                f.write(f'{time.time():.3f},' + r.stdout.strip() + '\n'); f.flush()
                done.wait(5)
    t=threading.Thread(target=monitor, daemon=True)
    p=None
    def stop(_sig, _frame):
        subprocess.run(['docker','stop','--time','10',container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        t.start()
        p=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (out / 'pipeline.log').open('w') as log:
            for line in p.stdout:
                # Also sanitize plugin diagnostics, which can include URI/auth material.
                for value in [camera.password, camera.uri]:
                    if value: line=line.replace(value, '<redacted>')
                log.write(line); log.flush(); print(line, end='', flush=True)
        return p.wait()
    finally:
        done.set();t.join(timeout=6)
        if p is not None and p.poll() is None: stop(None,None)
        secret.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())

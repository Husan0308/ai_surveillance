#!/usr/bin/env python3
"""Single-run 13-hour DeepStream soak supervisor (no service installation).

systemd transient user service owns the supervisor; ExecStopPost provides an
independent cleanup attempt after supervisor failure. Docker cleanup is only
performed for an exact container proven to belong to the recorded launcher.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / '.runtime/six-camera-soak'
MONITOR = ROOT / 'scripts/six_camera_live/monitor.sh'
VIEWER = ROOT / '.runtime/deepstream-rtsp-pause-suppression-reporting-20261030-r1/sources/apps/sample_apps/deepstream-app/deepstream-app'
MONITOR_HASH = 'fab0c7339ae627cd84cc6aa1b44e0b8919661767d04cfb72d5eafa2d551f0585'
VIEWER_HASH = '2602a28a191e3b8671e5e9e4d4fda1fc98d428b70b2d4235bfea8684c269483e'
UNIT = 'six-camera-soak-13h.service'
LIMIT = 46800
SYSTEMD_RUNTIME = 46500
NORMAL_STOP = 46420
SYSTEMD_STOP = 240
SAMPLE_SECONDS = 60
REPORT_MAX_BYTES = 32 * 1024 * 1024
STATE = RUNTIME / 'active.json'
LOCK = RUNTIME / 'control.lock'
CONTAINER_RE = re.compile(r'^ai-surveillance-six-live-(\d+)-[0-9a-f]{12}$')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def locked_control():
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    if RUNTIME.is_symlink() or RUNTIME.stat().st_uid != os.getuid():
        raise RuntimeError('unsafe runtime directory ownership')
    os.chmod(RUNTIME, 0o700)
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def load_state():
    if not STATE.exists():
        return None
    data = json.loads(STATE.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('run_id'), str):
        raise RuntimeError('invalid active state')
    return data


def save_state(data):
    p = RUNTIME / f'.state-{os.getpid()}-{secrets.token_hex(4)}.tmp'
    try:
        with p.open('x', encoding='utf8') as f:
            os.chmod(p, 0o600)
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(p, STATE)
    finally:
        p.unlink(missing_ok=True)


def append(report, record):
    path = Path(report)
    payload = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with path.open('ab', buffering=0) as f:
        if os.fstat(f.fileno()).st_size + len(payload) > REPORT_MAX_BYTES:
            raise RuntimeError('JSONL report size limit exceeded')
        f.write(payload)
        os.fsync(f.fileno())


def records(report):
    p = Path(report)
    if not p.exists():
        return []
    out = []
    with p.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                break
            try:
                out.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                break
    return out


def run_cmd(args, *, timeout=8):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def systemd_state():
    result = run_cmd(['systemctl', '--user', 'show', UNIT, '--property=ActiveState', '--value'], timeout=5)
    if result.returncode != 0:
        if 'not found' in result.stderr.lower() or 'not loaded' in result.stderr.lower():
            return 'inactive'
        raise RuntimeError('cannot inspect systemd user manager')
    return result.stdout.strip()


def proc_ticks(pid):
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def health(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.load(response)
            return response.status, data if isinstance(data, dict) else {}
    except (OSError, ValueError, TimeoutError):
        return None, {}


def docker_names():
    r = run_cmd(['docker', 'ps', '-a', '--format', '{{.Names}}'], timeout=7)
    if r.returncode != 0:
        raise RuntimeError('Docker inspection failed; cannot assume no containers')
    return r.stdout.splitlines()


def session_ready():
    display = os.environ.get('DISPLAY') or ':0'
    xauth = os.environ.get('XAUTHORITY') or f'/run/user/{os.getuid()}/gdm/Xauthority'
    if not Path(xauth).is_file():
        return False, display, xauth
    try:
        r = run_cmd(['xdpyinfo', '-display', display], timeout=5)
        return r.returncode == 0, display, xauth
    except (OSError, subprocess.TimeoutExpired):
        return False, display, xauth


def preflight():
    if digest(MONITOR) != MONITOR_HASH or digest(VIEWER) != VIEWER_HASH:
        raise RuntimeError('production hashes do not match approved versions')
    if any(CONTAINER_RE.fullmatch(n) for n in docker_names()):
        raise RuntimeError('existing six-camera container found; manual inspection required')
    ready, display, xauth = session_ready()
    if not ready:
        raise RuntimeError('authenticated X11 desktop session required')
    if health('http://127.0.0.1:8100/health')[0] != 200:
        raise RuntimeError('API is not healthy on port 8100')
    if health('http://127.0.0.1:8001/health')[0] != 200:
        raise RuntimeError('ML Service is not healthy on port 8001')
    return display, xauth


def owner_from_launcher(data):
    """Require matching PID/start-ticks in the existing monitor ownership lock."""
    lock = ROOT / '.runtime/six-live-preview.lock'
    try:
        with lock.open('r') as f:
            owner = json.load(f)
    except (OSError, ValueError):
        return None
    pid = data.get('monitor_pid')
    ticks = data.get('monitor_ticks')
    if owner.get('mode') != 'daily' or owner.get('pid') != pid or owner.get('start_ticks') != ticks:
        return None
    name = owner.get('container_name')
    if name and (not isinstance(name, str) or not CONTAINER_RE.fullmatch(name)
                 or not name.startswith(f'ai-surveillance-six-live-{pid}-')):
        return None
    return owner


def snapshot_owner(data):
    owner = owner_from_launcher(data)
    if owner is None:
        return None
    name = owner.get('container_name')
    if name and data.get('container_name') not in (None, name):
        raise RuntimeError('owned container identity changed')
    if name and data.get('container_name') is None:
        data['container_name'] = name
        save_state(data)
    return owner


def has_container(name):
    return name in docker_names() if name else False


def cleanup(data, *, reason, fallback=False):
    """Never signal/force-remove by broad image or container prefix alone."""
    verified = owner_from_launcher(data)
    name = data.get('container_name')
    pid, ticks = data.get('monitor_pid'), data.get('monitor_ticks')
    owner_match = verified is not None
    stop_rc = None
    if owner_match and proc_ticks(pid) == ticks:
        try:
            stop_rc = run_cmd(['bash', str(MONITOR), 'stop'], timeout=38).returncode
        except (OSError, subprocess.TimeoutExpired):
            stop_rc = 2
    elif proc_ticks(pid) == ticks and not owner_match:
        stop_rc = 2
    else:
        stop_rc = 0
    remaining = False
    docker_ok = True
    try:
        remaining = has_container(name)
        if remaining and owner_match:
            rm = run_cmd(['docker', 'rm', '-f', '--', name], timeout=12)
            docker_ok = rm.returncode == 0 and not has_container(name)
            remaining = not docker_ok
        elif remaining:
            docker_ok = False
    except (OSError, subprocess.TimeoutExpired, RuntimeError):
        docker_ok = False
    monitor_alive = proc_ticks(pid) == ticks and ticks is not None
    try:
        unfamiliar = [n for n in docker_names() if CONTAINER_RE.fullmatch(n) and n != name]
        if unfamiliar:
            docker_ok = False
    except RuntimeError:
        docker_ok = False
    lock_free = True
    lock_path = ROOT / '.runtime/six-live-preview.lock'
    if lock_path.exists():
        try:
            fd = os.open(lock_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                lock_free = False
            finally:
                os.close(fd)
        except OSError:
            lock_free = False
    ok = stop_rc in (0, None) and not monitor_alive and not remaining and docker_ok and lock_free
    return {'ok': ok, 'reason': reason, 'fallback': fallback, 'monitor_stop_rc': stop_rc,
            'monitor_alive': monitor_alive, 'owned_container_remaining': remaining,
            'docker_inspection_ok': docker_ok, 'launcher_lock_free': lock_free,
            'ownership_verified': owner_match}


def gpu_stats():
    try:
        r = run_cmd(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total',
                     '--format=csv,noheader,nounits'], timeout=5)
        if r.returncode != 0:
            return None
        values = [float(x.strip()) for x in r.stdout.splitlines()[0].split(',')]
        return dict(zip(('gpu_percent', 'temperature_c', 'vram_used_mib', 'vram_total_mib'), values))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def host_stats():
    try:
        fields = {}
        for row in Path('/proc/meminfo').read_text().splitlines():
            key, _, val = row.partition(':')
            if key in ('MemTotal', 'MemAvailable'):
                fields[key] = round(int(val.strip().split()[0]) / 1024, 2)
        load = [float(v) for v in Path('/proc/loadavg').read_text().split()[:3]]
        return {'ram_total_mib': fields.get('MemTotal'),
                'ram_available_mib': fields.get('MemAvailable'), 'loadavg': load}
    except (OSError, ValueError, IndexError):
        return None


def sample(data, elapsed):
    a_status, a = health('http://127.0.0.1:8100/health')
    m_status, m = health('http://127.0.0.1:8001/health')
    allowed = ('monitoring_status', 'analytics_status', 'six_camera_readiness', 'shared_pipeline_status')
    ml = {k: m.get(k) for k in allowed}
    ml.update(http=m_status)
    permitted = ('camera_id', 'online', 'current_state', 'decode_fps',
                 'tracker_fps', 'decoded_fps', 'source_fps', 'tracking_fps',
                 'age_since_last_decoded_frame_ms', 'age_since_last_pgie_frame_ms',
                 'age_since_last_tracker_output_ms', 'age_since_last_preview_ms')
    raw_cameras = m.get('cameras', [])
    ml['cameras'] = [{key: row[key] for key in permitted if key in row
                      and isinstance(row[key], (str, int, float, bool, type(None)))}
                     for row in raw_cameras if isinstance(row, dict)] if isinstance(raw_cameras, list) else []
    return {'type': 'health', 'utc': timestamp(), 'elapsed_seconds': round(elapsed, 2),
            'api_http': a_status, 'api_ml_accessible': a.get('ml_accessible'),
            'ml': ml, 'gpu': gpu_stats(), 'host': host_stats(),
            'monitor_alive': proc_ticks(data.get('monitor_pid')) == data.get('monitor_ticks')}


def supervise(run_id):
    data = load_state()
    if data is None or data['run_id'] != run_id:
        raise RuntimeError('run identity mismatch')
    if data['state'] != 'starting':
        raise RuntimeError('run not in starting state')
    report = data['report']
    shutdown_requested = {'reason': None}

    def on_signal(signum, _frame):
        shutdown_requested['reason'] = 'systemd_stop' if signum == signal.SIGTERM else 'interrupt'

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    started = time.monotonic()
    child = None
    rc = None
    last_sample = -SAMPLE_SECONDS
    last_hour = 0
    final_reason = 'unknown'
    try:
        env = os.environ.copy()
        with (Path(data['run_dir']) / 'launcher.log').open('ab', buffering=0) as logfile:
            child = subprocess.Popen(['bash', str(MONITOR), 'start'], cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=logfile,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            data.update(state='running', supervisor_pid=os.getpid(),
                        supervisor_ticks=proc_ticks(os.getpid()), monitor_pid=child.pid,
                        monitor_ticks=proc_ticks(child.pid))
            save_state(data)
            append(report, {'type': 'monitor_started', 'utc': timestamp(),
                            'pid': child.pid, 'monitor_ticks': data['monitor_ticks']})
            while True:
                elapsed = time.monotonic() - started
                snapshot_owner(data)
                rc = child.poll()
                if rc is not None:
                    final_reason = 'monitor_exited'
                    break
                if shutdown_requested['reason']:
                    final_reason = shutdown_requested['reason']
                    break
                if elapsed - last_sample >= SAMPLE_SECONDS and not session_ready()[0]:
                    final_reason = 'x11_session_lost'
                    break
                if elapsed >= NORMAL_STOP:
                    final_reason = 'automatic_deadline'
                    break
                if elapsed - last_sample >= SAMPLE_SECONDS:
                    row = sample(data, elapsed)
                    append(report, row)
                    last_sample = elapsed
                    hour = int(elapsed // 3600)
                    if hour > last_hour:
                        append(report, {'type': 'hourly', 'utc': timestamp(),
                                        'hour': hour, 'last_readiness': row['ml'].get('six_camera_readiness')})
                        last_hour = hour
                time.sleep(0.5)
    except BaseException as exc:
        final_reason = 'supervisor_exception'
        append(report, {'type': 'error', 'utc': timestamp(), 'kind': type(exc).__name__})
    if child is not None:
        result = cleanup(data, reason=final_reason)
        try:
            rc = child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            result['ok'] = False
    else:
        result = {'ok': False, 'reason': final_reason, 'failure': 'monitor_not_started'}
    good = result['ok'] and rc == 0 and final_reason in ('automatic_deadline', 'systemd_stop', 'interrupt')
    append(report, {'type': 'final', 'utc': timestamp(), 'reason': final_reason,
                    'monitor_exit': rc, 'cleanup': result,
                    'status': 'clean' if good else 'failed'})
    data.update(state='stopped' if good else 'failed', stop_reason=final_reason,
                monitor_exit=rc, finished_epoch=time.time())
    save_state(data)
    return 0 if good else 1


def post_stop(run_id):
    data = load_state()
    if data is None or data['run_id'] != run_id:
        return 2
    rows = records(data['report'])
    if any(r.get('type') == 'final' for r in rows):
        return 0
    result = cleanup(data, reason='supervisor_lost', fallback=True)
    append(data['report'], {'type': 'final', 'utc': timestamp(),
                            'reason': 'supervisor_lost', 'cleanup': result, 'status': 'failed'})
    data.update(state='failed', stop_reason='supervisor_lost', finished_epoch=time.time())
    save_state(data)
    return 0 if result['ok'] else 2


def start():
    fd = locked_control()
    try:
        st = load_state()
        if systemd_state() in ('active', 'activating', 'deactivating', 'reloading'):
            raise RuntimeError('soak systemd unit is already active')
        if st and st.get('state') in ('running', 'starting'):
            raise RuntimeError('previous soak has unresolved state; manual inspection required')
        display, xauth = preflight()
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(5)
        run_dir = RUNTIME / run_id
        run_dir.mkdir(mode=0o700)
        report = run_dir / 'monitor_13h.jsonl'
        report.touch(mode=0o600, exist_ok=False)
        data = {'run_id': run_id, 'state': 'starting', 'run_dir': str(run_dir),
                'report': str(report), 'started_epoch': time.time(),
                'deadline_epoch': time.time() + LIMIT,
                'viewer_sha256': VIEWER_HASH, 'monitor_sha256': MONITOR_HASH,
                'unit': UNIT}
        save_state(data)
        append(report, {'type': 'start', 'utc': timestamp(), 'run_id': run_id,
                        'limit_seconds': LIMIT, 'viewer_sha256': VIEWER_HASH,
                        'monitor_sha256': MONITOR_HASH})
        exe = sys.executable
        command = [
            'systemd-run', '--user', '--collect', f'--unit={UNIT}',
            '--property=Type=exec', '--property=Restart=no',
            '--property=KillMode=mixed',
            f'--property=RuntimeMaxSec={SYSTEMD_RUNTIME}s',
            f'--property=TimeoutStopSec={SYSTEMD_STOP}s',
            '--property=WorkingDirectory=' + str(ROOT),
            '--property=ExecStopPost=' + ' '.join(shlex.quote(x) for x in
                (exe, '-B', str(Path(__file__).resolve()), 'post-stop', run_id)),
            '--setenv=DISPLAY=' + display, '--setenv=XAUTHORITY=' + xauth,
            exe, '-B', str(Path(__file__).resolve()), 'run', run_id,
        ]
        result = run_cmd(command, timeout=15)
        if result.returncode != 0:
            append(report, {'type': 'error', 'utc': timestamp(),
                            'kind': 'systemd_start_failed', 'message': result.stderr[-400:]})
            data.update(state='failed', finished_epoch=time.time())
            save_state(data)
            raise RuntimeError('transient systemd service launch failed: ' + result.stderr[-200:])
        print(f'START REQUESTED | unit={UNIT}\nReport: {report}\nUse status to confirm running.')
    finally:
        os.close(fd)


def stop():
    fd = locked_control()
    try:
        data = load_state()
        if not data or data.get('state') not in ('starting', 'running'):
            raise RuntimeError('no active soak run')
        if systemd_state() not in ('active', 'activating'):
            raise RuntimeError('systemd unit not active; inspect orphan ownership manually')
        r = run_cmd(['systemctl', '--user', 'stop', UNIT], timeout=SYSTEMD_STOP + 15)
        if r.returncode != 0:
            raise RuntimeError('systemd stop failed; inspect report and unit journal')
        after = load_state()
        print(f'STOP requested | state={after.get("state")} | report={after.get("report")}')
    finally:
        os.close(fd)


def status():
    data = load_state()
    if not data:
        print('No run has been started')
        return
    try:
        active = systemd_state()
    except (OSError, RuntimeError):
        active = 'unknown'
    elapsed = max(0, time.time() - data['started_epoch'])
    state = data.get('state')
    if state in ('running', 'starting') and active not in ('active', 'activating'):
        state = 'UNVERIFIED_OR_ORPHANED'
    rows = records(data['report'])
    latest = next((r for r in reversed(rows) if r.get('type') == 'health'), None)
    print(json.dumps({'state': state, 'unit_state': active, 'elapsed_seconds': int(elapsed),
                      'remaining_seconds': max(0, int(LIMIT - elapsed)),
                      'report': data['report'], 'latest_health': latest}, indent=2))


def report():
    data = load_state()
    if not data:
        print('No saved run')
        return
    rows = records(data['report'])
    kinds = {}
    for row in rows:
        kinds[row.get('type', 'unknown')] = kinds.get(row.get('type', 'unknown'), 0) + 1
    print(json.dumps({'file': data['report'], 'records': len(rows),
                      'kinds': kinds, 'last': rows[-1] if rows else None}, indent=2))


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: soak_13h.sh {start|stop|status|report}')
    mode = sys.argv[1]
    if mode == 'run' and len(sys.argv) == 3:
        return supervise(sys.argv[2])
    if mode == 'post-stop' and len(sys.argv) == 3:
        return post_stop(sys.argv[2])
    if len(sys.argv) != 2:
        raise SystemExit('unexpected arguments')
    if mode not in ('start', 'stop', 'status', 'report'):
        raise SystemExit('usage: soak_13h.sh {start|stop|status|report}')
    try:
        {'start': start, 'stop': stop, 'status': status, 'report': report}[mode]()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'SOAK FAILED: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

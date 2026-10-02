#!/usr/bin/env python3
"""Isolated, one-factor replay against the recorded effective V8 baseline.

Prefixes are decoded-byte/PTS checked and diagnostic only. Full acceptance must
use the original complete inputs. The archived runner and evidence stay intact.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / '.runtime/final-detector-ab-20261002-uSpPqjLY'
CLEAN = ROOT / '.runtime/gate3jr-DV3lTHm8/recovery'
BASELINE = ARCHIVE / 'runs/nvdcf-retention-v8-canonical/replay-20261002-181509'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decoded_frames(path, count):
    output = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                      '-map', '0:v:0', '-frames:v', str(count),
                                      '-f', 'framemd5', '-'], text=True)
    return [line.strip() for line in output.splitlines() if not line.startswith('#')]


def verify_prefix(original, copied, count):
    if len(original) != count or len(copied) != count or original != copied:
        raise ValueError('diagnostic prefix differs in decoded frames or timestamps')


def prepare_inputs(stage, dataset, prefix):
    inputs = json.loads((ARCHIVE / 'inputs.json').read_text())
    if not prefix:
        return inputs, None
    if dataset != 'canonical':
        raise ValueError('diagnostic prefixes are canonical only')
    proof = {}
    videos = stage / 'prefix-videos'
    videos.mkdir()
    for camera, item in inputs[dataset].items():
        source = Path(item['path'])
        assert sha(source) == item['sha256']
        target = videos / source.name
        # Stream-copy truncation can retain a future reference picture and lose
        # the final B-frame. Use lossless 4:2:0 H.264 and verify every decoded
        # byte and PTS instead of quietly accepting that boundary corruption.
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                        '-map', '0:v:0', '-frames:v', str(prefix), '-c:v', 'libx264',
                        '-qp', '0', '-pix_fmt', 'yuv420p', '-bf', '0', '-g', '40', '-preset', 'fast',
                        str(target)], check=True)
        original = decoded_frames(source, prefix)
        copied = decoded_frames(target, prefix)
        verify_prefix(original, copied, prefix)
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration',
            '-of', 'json', str(target)], text=True))
        assert int(probe['streams'][0]['nb_read_frames']) == prefix
        proof[camera] = dict(original_source=str(source), original_sha256=item['sha256'],
                             prefix_sha256=sha(target), frames=prefix,
                             decoded_bytes_and_pts_equal=True, encoding='lossless_H264_no_B_frames_diagnostic_only',
                             framemd5_sha256=hashlib.sha256('\n'.join(original).encode()).hexdigest())
        item.update(path=str(target.resolve()), sha256=sha(target), probe=probe)
    return inputs, proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', type=Path, required=True, help='New directory under .runtime')
    ap.add_argument('--label', required=True)
    ap.add_argument('--dataset', choices=['canonical', 'person-present', 'empty-room'], default='canonical')
    ap.add_argument('--parameter')
    ap.add_argument('--value', type=float)
    ap.add_argument('--baseline-run', type=Path, default=BASELINE,
                    help='Recorded experimental parent; immutable V8 remains default')
    ap.add_argument('--prefix-frames', type=int)
    ap.add_argument('--identity-diagnostics', action='store_true',
                    help='Snapshot experimental manager/worker and enable decision context')
    args = ap.parse_args()
    stage = args.stage.resolve()
    stage.relative_to((ROOT / '.runtime').resolve())
    stage.mkdir(parents=True, exist_ok=False)
    inputs, proof = prepare_inputs(stage, args.dataset, args.prefix_frames)
    (stage / 'inputs.json').write_text(json.dumps(inputs, indent=2))
    sys.path[:0] = [str(CLEAN), str(CLEAN / 'scripts/dev_room_mv3dt'), str(ROOT)]
    import run_room_pair as runner
    if args.identity_diagnostics:
        worker = stage / 'diagnostic_identity_worker.py'
        manager = stage / 'global_identity_manager.py'
        manager.write_bytes((ROOT / 'services/mv3dt_room/global_identity_manager.py').read_bytes())
        live_worker = stage / 'live_identity_worker.py'
        live_worker.write_bytes((ROOT / 'services/mv3dt_room/live_identity_worker.py').read_bytes())
        worker.write_text(
            'import importlib.util, os, runpy, sys\n'
            f'sys.path[:0] = [{str(CLEAN)!r}, {str(CLEAN / "services/mv3dt_room")!r}]\n'
            'os.environ["MV3DT_IDENTITY_DECISION_DETAILS"] = "1"\n'
            'name = "services.mv3dt_room.global_identity_manager"\n'
            f'spec = importlib.util.spec_from_file_location(name, {str(manager)!r})\n'
            'module = importlib.util.module_from_spec(spec)\n'
            'sys.modules[name] = module\n'
            'spec.loader.exec_module(module)\n'
            f'runpy.run_path({str(live_worker)!r}, run_name="__main__")\n')
        # Validate the worker's import environment before starting a GPU run.
        subprocess.run([str(runner.OSNET_PYTHON), str(worker), '--help'], check=True,
                       stdout=subprocess.DEVNULL)
        original_popen = subprocess.Popen
        archived_worker = str(CLEAN / 'services/mv3dt_room/live_identity_worker.py')
        def diagnostic_popen(command, *a, **kw):
            if isinstance(command, list) and archived_worker in command:
                command = [str(worker) if item == archived_worker else item for item in command]
            return original_popen(command, *a, **kw)
        subprocess.Popen = diagnostic_popen
    spec = importlib.util.spec_from_file_location('recorded_v8_runner', ARCHIVE / 'run_candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STAGE = stage
    baseline_run = args.baseline_run.resolve()
    baseline_run.relative_to((ROOT / '.runtime').resolve())
    assert sha(baseline_run / 'run/config_pgie.txt') == sha(BASELINE / 'run/config_pgie.txt')
    baseline = module.scalars((baseline_run / 'run/config_tracker.yml').read_text())
    # Adopt only the already recorded experimental tracker scalar policy.
    # Transport/output diagnostics are supplied identically by the harness.
    production = module.scalars((runner.PROFILE / 'config_tracker.yml').read_text())
    for key, value in baseline.items():
        if key in production and key.startswith(('TargetManagement.', 'DataAssociator.', 'PoseEstimator.')):
            if key == 'TargetManagement.outputShadowTracks':
                continue
            if value != production[key]:
                module.YOLO_DELTA[key] = float(value)
    if args.parameter:
        assert args.parameter in baseline and args.value is not None
        module.YOLO_DELTA[args.parameter] = args.value
    protected = {str(ROOT / 'config/mv3dt_dev_room' / n): sha(ROOT / 'config/mv3dt_dev_room' / n)
                 for n in ['config_tracker.yml', 'config_deepstream.txt', 'config_pgie.txt']}
    provenance = dict(parameter=args.parameter, value=args.value,
                      original_runner_sha256=sha(ARCHIVE / 'run_candidate.py'),
                      effective_v8_tracker_sha256=sha(BASELINE / 'run/config_tracker.yml'),
                      experimental_parent=str(baseline_run),
                      experimental_parent_tracker_sha256=sha(baseline_run / 'run/config_tracker.yml'),
                      identity_diagnostics=args.identity_diagnostics,
                      identity_manager_sha256=sha(stage / 'global_identity_manager.py') if args.identity_diagnostics else None,
                      identity_worker_sha256=sha(stage / 'live_identity_worker.py') if args.identity_diagnostics else None,
                      prefix_is_diagnostic_only=bool(args.prefix_frames), prefix_proof=proof,
                      protected_before=protected)
    (stage / 'causal-provenance.json').write_text(json.dumps(provenance, indent=2))
    sys.argv = [str(ARCHIVE / 'run_candidate.py'), 'yolo26m', '--dataset', args.dataset, '--label', args.label]
    try:
        module.main()
    finally:
        assert all(sha(p) == h for p, h in protected.items())
        (stage / 'protected-after.json').write_text(json.dumps(protected, indent=2))
    folder = stage / 'runs' / args.label
    run = Path(json.loads((folder / 'completed.json').read_text())['run'])
    actual = module.scalars((run / 'run/config_tracker.yml').read_text())
    normalize = lambda value: float(value) if value.replace('.', '', 1).isdigit() else value
    delta = {k: [v, actual[k]] for k, v in baseline.items() if normalize(v) != normalize(actual[k])}
    assert set(delta) == ({args.parameter} if args.parameter else set()), delta
    provenance.update(actual_tracker_delta=delta, tracker_sha256=sha(run / 'run/config_tracker.yml'),
                      detector_sha256=sha(run / 'run/config_pgie.txt'))
    (stage / 'causal-provenance.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()

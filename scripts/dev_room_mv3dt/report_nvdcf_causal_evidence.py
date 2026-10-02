#!/usr/bin/env python3
"""Expand archived canonical diagnostics without changing acceptance metrics.

Observed SDK state and image geometry are evidence, not private NvDCF scores.
Only the explicitly reviewed creation frames carry physical-person labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev_room_mv3dt.diagnose_canonical_retention import iou, jsonl, public_lifecycle


def xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def summarize_window(camera, window, details, audit, lifecycle=None):
    rows = [d for d in details if d['proposal']['camera'] == camera
            and window['start'] <= d['proposal']['frame'] <= window['end']]
    frames = []
    for frame in range(window['start'], window['end'] + 1):
        stages = audit[camera, frame]
        losses = [dict(d) for d in rows if d['proposal']['frame'] == frame]
        if lifecycle is not None:
            for loss in losses:
                for direction in ('before', 'after'):
                    outputs = loss[f'nearest_spatial_output_{direction}']
                    loss[f'lifecycle_{direction}'] = [
                        sample for output in outputs
                        for sample in lifecycle.get((camera, output['frame']), [])
                        if sample['native'] == output['native']]
                loss['published_state_not_exported'] = 'Do not infer ACTIVE from absence of SDK state'
        frames.append(dict(
            frame=frame,
            detector_object_count=len(stages.get('pgie', [])),
            tracker_object_count=len(stages.get('tracker', [])),
            detector_confidences=[o['confidence'] for o in stages.get('pgie', [])],
            unmatched_proposals=losses,
        ))
    return dict(camera=camera, start_frame=window['start'], end_frame=window['end'],
                duration_frames=window['frames'], frames=frames,
                suppression='NOT_EXPORTED_BY_SDK',
                private_association_rejection='NOT_EXPORTED_BY_SDK',
                physical_person='See reviewed source overlays; geometry alone does not prove identity')


def identity_timeline(summary, published):
    timelines = []
    for creation in summary['identity_creations']:
        event = creation['event']
        trace = creation['decision_traces'][-1] if creation['decision_traces'] else {}
        obs = trace.get('observation', {})
        candidates = []
        for item in trace.get('candidate_trace', []):
            same_camera = [p for p in published if p['camera_id'] == event['camera_id']
                           and p['global_person_id'] == item['application_id']
                           and p['frame'] <= event['frame']]
            previous = max(same_camera, key=lambda p: p['frame'], default=None)
            candidates.append(dict(
                candidate=item['application_id'],
                reid_similarity=item['cosine_per_camera'],
                best_gallery_similarity=item['cosine_best_gallery'],
                rejection=item['exact_rejection_reason'],
                time_delta_ms=item['timestamp_difference_ms'],
                native_mv3dt_shared=item['native_mva_continuity_evidence'],
                cross_camera_evidence=item['cross_camera_continuity_evidence'],
                geometry_distance=item['world_distance_m'],
                predecessor=previous,
                predecessor_frame_gap=event['frame'] - previous['frame'] if previous else None,
                predecessor_bbox_iou=iou(xywh(obs['bbox']), xywh(previous['bbox'])) if previous else None,
                accepted_candidate_score=item['final_candidate_score'],
            ))
        timelines.append(dict(
            canonical=f"Person_{event['global_person_id']:02d}",
            first_frame=event['frame'], camera=event['camera_id'],
            native_tracker_id=event['native_track_id'],
            native_first_output=creation['first_native_output'],
            creation_reason=event['reason'], novelty=trace.get('novelty'),
            candidates=candidates,
            physical_person='REQUIRES_SOURCE_REVIEW',
        ))
    return timelines


def report(diagnosis, output):
    diagnosis, output = Path(diagnosis), Path(output)
    summary = json.loads((diagnosis / 'diagnostic-summary.json').read_text())
    run = Path(summary['run'])
    details = jsonl(diagnosis / 'unmatched-proposals.jsonl')
    audit = defaultdict(dict)
    for row in jsonl(run / 'run/logs/probe/frame_path_audit.jsonl'):
        if row.get('record') == 'frame' and row.get('stage') in ('pgie', 'tracker'):
            audit[row['mapped_camera_id'], row['frame_num']][row['stage']] = row['objects']
    lifecycle = public_lifecycle(run / 'run/logs/probe')
    windows = [summarize_window(cam, w, details, audit, lifecycle)
               for cam, camera in summary['cameras'].items() for w in camera['deficit_windows']]
    published = jsonl(run / 'identity-live/global_identity.jsonl')
    timelines = identity_timeline(summary, published)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'all-deficit-windows.json').write_text(json.dumps(windows, indent=2))
    (output / 'identity-creation-timeline.json').write_text(json.dumps(timelines, indent=2))
    (output / 'native-to-canonical-trajectories.json').write_text(json.dumps(summary['local_to_canonical'], indent=2))
    (output / 'summary.json').write_text(json.dumps(dict(
        run=str(run), metric_changes=False, window_count=len(windows),
        camera_metrics=summary['cameras'], identity_creation_count=len(timelines),
        unavailable_private_fields=summary['unavailable_private_fields'],
        physical_scope='Creation frames reviewed separately; native mappings are observations, not physical ground truth',
    ), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report(args.diagnosis, args.output)


if __name__ == '__main__':
    main()

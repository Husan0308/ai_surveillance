import importlib
import json
from pathlib import Path

from services.mv3dt_room.identity_metrics import IdentityMetrics


def worker_without_inference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'services/mv3dt_room'))
    cls = importlib.import_module('services.mv3dt_room.live_identity_worker').LiveIdentityWorker
    worker = cls.__new__(cls)
    worker.kafka_offset = 0
    worker.metrics = IdentityMetrics()
    received = []
    worker._ingest_frame = lambda camera, frame, *_: received.append((camera, frame))
    return worker, received


def message(frame):
    return json.dumps({'frame': {'id': frame, 'sensorId': 'CAM-01'}}) + '\n'


def test_partial_capture_line_is_received_exactly_once_after_completion(tmp_path, monkeypatch):
    worker, received = worker_without_inference(monkeypatch)
    path = tmp_path / 'capture.jsonl'
    first, second = message(1), message(2)
    boundary = len(second) // 2
    path.write_text(first + second[:boundary])
    assert worker._parse_new_kafka(path) == 1
    assert received == [('CAM-01', 1)]
    assert worker._parse_new_kafka(path) == 0
    with path.open('a') as stream:
        stream.write(second[boundary:])
    assert worker._parse_new_kafka(path) == 1
    assert received == [('CAM-01', 1), ('CAM-01', 2)]
    assert worker._parse_new_kafka(path) == 0
    assert worker.metrics.snapshot()['counters'].get('kafka_parse_errors', 0) == 0


def test_complete_malformed_record_still_counts_as_a_real_error(tmp_path, monkeypatch):
    worker, received = worker_without_inference(monkeypatch)
    path = tmp_path / 'capture.jsonl'
    path.write_text('malformed\n' + message(1))
    assert worker._parse_new_kafka(path) == 2
    assert received == [('CAM-01', 1)]
    assert worker.metrics.snapshot()['counters']['kafka_parse_errors'] == 1

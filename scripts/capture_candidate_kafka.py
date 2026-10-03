"""Capture only a fresh experimental topic; use the existing Frame schema."""
import argparse
import json
import signal
import time

from google.protobuf.json_format import MessageToDict
from kafka import KafkaConsumer, TopicPartition
from services.mv3dt_room.schema_pb2 import Frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    args = parser.parse_args()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092", enable_auto_commit=False,
                             consumer_timeout_ms=1000)
    partition = TopicPartition(args.topic, 0)
    consumer.assign([partition])
    start = consumer.end_offsets([partition])[partition]
    consumer.seek(partition, start)
    print(json.dumps({"event": "capture_start", "topic": args.topic, "offset": start,
                      "host_time_ns": time.time_ns()}), flush=True)
    count = 0
    while running:
        for records in consumer.poll(timeout_ms=250, max_records=1000).values():
            for message in records:
                frame = Frame()
                frame.ParseFromString(message.value)
                print(json.dumps({"event": "message", "offset": message.offset,
                    "broker_timestamp_ms": message.timestamp, "host_time_ns": time.time_ns(),
                    "frame": MessageToDict(frame)}, separators=(",", ":")), flush=True)
                count += 1
    print(json.dumps({"event": "capture_stop", "count": count, "host_time_ns": time.time_ns()}), flush=True)
    consumer.close()


if __name__ == "__main__":
    main()

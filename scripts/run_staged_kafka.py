"""Start a new F5 test broker without altering any existing broker storage.

Only used when localhost:9092/9093 are unoccupied. The official installed Kafka
configuration is copied, changing ONLY log.dirs to a new evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kafka-root", type=Path, default=Path("/home/apsidal/kafka_2.13-4.2.0"))
    args = parser.parse_args()
    for port in (9092, 9093):
        with socket.socket() as probe:
            probe.bind(("", port))  # Fail closed; never stop an existing broker.
    output = args.output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    original = args.kafka_root / "config/server.properties"
    text = original.read_text()
    new, count = re.subn(r"(?m)^log\.dirs=.*$", f"log.dirs={output / 'storage'}", text)
    if count != 1:
        raise ValueError("expected exactly one existing broker log.dirs setting")
    config = output / "server.properties"
    config.write_text(new)
    env = dict(os.environ, LOG_DIR=str(output / "service-logs"))
    uid = subprocess.check_output([str(args.kafka_root / "bin/kafka-storage.sh"), "random-uuid"], env=env, text=True).strip()
    command = [str(args.kafka_root / "bin/kafka-storage.sh"), "format", "--standalone", "-t", uid, "-c", str(config)]
    with (output / "format.log").open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "provenance.json").write_text(json.dumps({"source_config": str(original),
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "staged_sha256": hashlib.sha256(new.encode()).hexdigest(), "only_changed_property": "log.dirs",
        "cluster_id": uid, "format_command": command, "existing_storage_formatted": False}, indent=2))
    os.execve(str(args.kafka_root / "bin/kafka-server-start.sh"),
        [str(args.kafka_root / "bin/kafka-server-start.sh"), str(config)], env)


if __name__ == "__main__":
    main()

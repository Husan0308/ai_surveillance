from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from services.mv3dt_staging.event_store import sha256_file


TOPIC_CAMERA = {
    "/trck/cam1": ("CAM-02", 0),
    "/trck/cam2": ("CAM-05", 1),
}


class AdoptionAliasError(ValueError):
    """An adoption sidecar cannot be applied without risking ambiguous labels."""


@dataclass(frozen=True)
class AdoptionAlias:
    camera_id: str
    source_id: int
    raw_id: str
    display_id: str
    valid_from_frame: int
    valid_until_frame: int
    adoption_frame: int
    old_birth_timestamp_ns: int
    target_camera_id: str
    target_source_id: int
    target_birth_timestamp_ns: int
    event_timestamp_ns: int

    @property
    def source_key(self) -> tuple[str, int, str, int]:
        return (self.camera_id, self.source_id, self.raw_id, self.old_birth_timestamp_ns)

    @property
    def target_key(self) -> tuple[str, int, str, int]:
        return (self.target_camera_id, self.target_source_id, self.display_id,
                self.target_birth_timestamp_ns)


def _required_int(data: dict[str, Any], name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool):
        raise AdoptionAliasError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AdoptionAliasError(f"{name} must be an integer") from exc
    return result


def _topic_from_peer_message(message: dict[str, Any]) -> str | None:
    origin = message.get("fromCamId")
    if not isinstance(origin, str) or ";" not in origin:
        return None
    return origin.rsplit(";", 1)[1]


def build_bound_manifest(
    alias_map_path: str | Path,
    mqtt_decoded_path: str | Path,
    event_log_path: str | Path,
    mqtt_raw_path: str | Path,
) -> dict[str, Any]:
    """Bind verified ADOPTED_ID records to the exact offline Kafka and MQTT evidence."""
    alias_document = json.loads(Path(alias_map_path).read_text(encoding="utf-8"))
    if alias_document.get("scope") != "private offline replay presentation only":
        raise AdoptionAliasError("alias source is not scoped to offline review")
    aliases = alias_document.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        raise AdoptionAliasError("alias source contains no verified adoption records")

    mqtt_messages = []
    for line_number, line in enumerate(Path(mqtt_decoded_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdoptionAliasError(f"invalid decoded MQTT JSON at line {line_number}") from exc
        if isinstance(message, dict):
            mqtt_messages.append(message)

    bound_aliases: list[dict[str, Any]] = []
    for item in aliases:
        if not isinstance(item, dict):
            raise AdoptionAliasError("alias records must be JSON objects")
        raw_event = item.get("raw_event")
        if not isinstance(raw_event, dict):
            raise AdoptionAliasError("an alias is missing its preserved raw ADOPTED_ID event")
        adoption_fields = ("type", "targetId", "adoptedId", "frameNum", "targetIDTimeStamp", "messageTimeStamp")
        matching_events = [message for message in mqtt_messages
                           if all(message.get(key) == raw_event.get(key) for key in adoption_fields)]
        if len(matching_events) != 1 or raw_event.get("type") != 3:
            raise AdoptionAliasError("ADOPTED_ID record is absent or ambiguous in decoded MQTT evidence")
        if matching_events[0].get("fromCamId") != raw_event.get("fromCamId"):
            raise AdoptionAliasError("ADOPTED_ID topic provenance does not match the raw event")

        source_topic = item.get("mqtt_topic")
        if source_topic not in TOPIC_CAMERA:
            raise AdoptionAliasError(f"unsupported adoption source topic: {source_topic!r}")
        source_camera, source_id = TOPIC_CAMERA[source_topic]
        peer_id = str(item.get("display_id_after_adoption", ""))
        raw_id_before = str(item.get("raw_id_before_adoption", ""))
        old_birth = _required_int(item, "old_target_track_birth_timestamp_ns")
        target_birth = _required_int(item, "event_targetIDTimeStamp_ns")
        source_tracklets = [message for message in mqtt_messages
                            if message.get("type") == 1
                            and str(message.get("targetId")) == raw_id_before
                            and message.get("targetIDTimeStamp") == old_birth
                            and _topic_from_peer_message(message) == source_topic]
        if not source_tracklets:
            raise AdoptionAliasError("old target identity is not present in the source camera TRACKLET evidence")
        peer_tracklets = [message for message in mqtt_messages
                          if message.get("type") == 1
                          and str(message.get("targetId")) == peer_id
                          and message.get("targetIDTimeStamp") == target_birth]
        expected_matches = int(item.get("adopted_peer_tracklet_matches", -1))
        topic_counts: dict[str, int] = {}
        for message in peer_tracklets:
            topic = _topic_from_peer_message(message)
            if topic:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        peer_topics = [topic for topic, count in topic_counts.items()
                       if topic != source_topic and count == expected_matches]
        if len(peer_topics) != 1:
            raise AdoptionAliasError("adopted peer tracklet camera is missing or ambiguous")
        target_topic = peer_topics[0]
        if target_topic not in TOPIC_CAMERA:
            raise AdoptionAliasError(f"unsupported adopted peer topic: {target_topic!r}")
        target_camera, target_source_id = TOPIC_CAMERA[target_topic]
        if topic_counts[target_topic] != expected_matches:
            raise AdoptionAliasError("peer TRACKLET count differs from the preserved verified alias record")

        raw_target_id = str(raw_event.get("targetId", ""))
        adopted_id = str(raw_event.get("adoptedId", ""))
        adoption_frame = _required_int(item, "adoption_frame")
        if raw_target_id != str(item.get("raw_id_before_adoption")) or adopted_id != peer_id:
            raise AdoptionAliasError("alias IDs do not match the preserved ADOPTED_ID payload")
        if raw_event.get("frameNum") != adoption_frame:
            raise AdoptionAliasError("alias frame does not match the adoption event frame")

        bound_aliases.append({
            "camera_id": source_camera,
            "source_id": source_id,
            "raw_id": raw_target_id,
            "display_id": adopted_id,
            "valid_from_frame": _required_int(item, "display_alias_valid_from_frame"),
            "valid_until_frame": _required_int(item, "display_alias_valid_until_frame_inclusive"),
            "adoption_frame": adoption_frame,
            "old_birth_timestamp_ns": old_birth,
            "target_camera_id": target_camera,
            "target_source_id": target_source_id,
            "target_birth_timestamp_ns": target_birth,
            "event_timestamp_ns": _required_int(item, "event_message_timestamp_ns"),
            "event_type": "ADOPTED_ID",
            "peer_tracklet_matches": len(peer_tracklets),
            "mqtt_topic": source_topic,
            "target_topic": target_topic,
            "raw_event": raw_event,
        })

    return {
        "schema_version": 1,
        "scope": "offline_replay_only",
        "source_log_sha256": sha256_file(event_log_path),
        "mqtt_raw_sha256": sha256_file(mqtt_raw_path),
        "alias_map_sha256": sha256_file(alias_map_path),
        "aliases": bound_aliases,
    }


class AdoptionAliasIndex:
    """Validated offline adoption lineage; never changes source observations."""

    def __init__(self, manifest: dict[str, Any], *, expected_source_log_sha256: str,
                 camera_mapping: dict[str, tuple[str, int]]) -> None:
        if manifest.get("schema_version") != 1 or manifest.get("scope") != "offline_replay_only":
            raise AdoptionAliasError("adoption aliases must use schema 1 and offline_replay_only scope")
        if manifest.get("source_log_sha256") != expected_source_log_sha256:
            raise AdoptionAliasError("adoption alias source-log hash does not match the imported event database")
        aliases = manifest.get("aliases")
        if not isinstance(aliases, list):
            raise AdoptionAliasError("adoption aliases must be a list")
        mapping_by_camera = {camera: source_id for _sensor, (camera, source_id) in camera_mapping.items()}
        rules: list[AdoptionAlias] = []
        for item in aliases:
            if not isinstance(item, dict) or item.get("event_type") != "ADOPTED_ID":
                raise AdoptionAliasError("every alias must be backed by an ADOPTED_ID event")
            camera_id = str(item.get("camera_id", ""))
            source_id = _required_int(item, "source_id")
            target_camera_id = str(item.get("target_camera_id", ""))
            target_source_id = _required_int(item, "target_source_id")
            if camera_id not in mapping_by_camera or mapping_by_camera[camera_id] != source_id:
                raise AdoptionAliasError("alias source camera/source_id disagrees with the imported camera mapping")
            if (target_camera_id not in mapping_by_camera
                    or mapping_by_camera[target_camera_id] != target_source_id):
                raise AdoptionAliasError("alias target camera/source_id disagrees with the imported camera mapping")
            raw_id, display_id = str(item.get("raw_id", "")), str(item.get("display_id", ""))
            if not raw_id or not display_id or raw_id == display_id:
                raise AdoptionAliasError("alias must map two distinct non-empty IDs")
            start = _required_int(item, "valid_from_frame")
            end = _required_int(item, "valid_until_frame")
            adoption_frame = _required_int(item, "adoption_frame")
            if start < 0 or end < start or adoption_frame != end + 1:
                raise AdoptionAliasError("offline alias frame range must end immediately before adoption")
            old_birth = _required_int(item, "old_birth_timestamp_ns")
            target_birth = _required_int(item, "target_birth_timestamp_ns")
            event_timestamp = _required_int(item, "event_timestamp_ns")
            raw_event = item.get("raw_event")
            if (old_birth <= 0 or target_birth <= 0 or event_timestamp <= 0
                    or not isinstance(raw_event, dict)
                    or raw_event.get("type") != 3
                    or str(raw_event.get("targetId")) != raw_id
                    or str(raw_event.get("adoptedId")) != display_id
                    or raw_event.get("frameNum") != adoption_frame
                    or raw_event.get("targetIDTimeStamp") != target_birth
                    or raw_event.get("messageTimeStamp") != event_timestamp):
                raise AdoptionAliasError("alias fields do not match the preserved ADOPTED_ID payload")
            rules.append(AdoptionAlias(
                camera_id, source_id, raw_id, display_id, start, end, adoption_frame,
                old_birth, target_camera_id, target_source_id, target_birth, event_timestamp,
            ))

        if not rules:
            raise AdoptionAliasError("adoption manifest has no usable aliases")
        self._validate_rules(rules)
        self.rules = tuple(rules)
        self._by_source_key = {rule.source_key: rule for rule in rules}
        self._by_source_id = {}
        for rule in rules:
            self._by_source_id.setdefault((rule.camera_id, rule.source_id, rule.raw_id), []).append(rule)
        self.source_log_sha256 = expected_source_log_sha256
        self.evidence_mqtt_sha256 = str(manifest.get("mqtt_raw_sha256", ""))

    @classmethod
    def from_file(cls, path: str | Path, *, expected_source_log_sha256: str,
                  camera_mapping: dict[str, tuple[str, int]]) -> "AdoptionAliasIndex":
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdoptionAliasError(f"cannot read adoption alias manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise AdoptionAliasError("adoption alias manifest must be a JSON object")
        return cls(manifest, expected_source_log_sha256=expected_source_log_sha256,
                   camera_mapping=camera_mapping)

    @staticmethod
    def _validate_rules(rules: list[AdoptionAlias]) -> None:
        by_source: dict[tuple[str, int, str, int], AdoptionAlias] = {}
        for rule in rules:
            if rule.source_key in by_source:
                raise AdoptionAliasError("conflicting duplicate adoption source identity")
            by_source[rule.source_key] = rule
        # Frame records do not contain tracker birth IDs, so overlapping aliases for
        # the same camera/raw ID cannot be safely distinguished and are rejected.
        for index, left in enumerate(rules):
            for right in rules[index + 1:]:
                overlap = max(left.valid_from_frame, right.valid_from_frame) <= min(
                    left.valid_until_frame, right.valid_until_frame
                )
                if not overlap:
                    continue
                if ((left.camera_id, left.source_id, left.raw_id)
                        == (right.camera_id, right.source_id, right.raw_id)):
                    raise AdoptionAliasError("overlapping aliases make a raw camera track ambiguous")
                if ((left.camera_id, left.source_id, left.display_id)
                        == (right.camera_id, right.source_id, right.display_id)):
                    raise AdoptionAliasError("overlapping aliases merge distinct same-camera tracks")
                if left.target_key == right.target_key:
                    raise AdoptionAliasError("overlapping aliases create an ambiguous cross-camera identity merge")

        visiting: set[tuple[str, int, str, int]] = set()
        visited: set[tuple[str, int, str, int]] = set()

        def visit(key: tuple[str, int, str, int]) -> None:
            if key in visiting:
                raise AdoptionAliasError("cycle detected in ADOPTED_ID alias lineage")
            if key in visited:
                return
            visiting.add(key)
            rule = by_source.get(key)
            if rule is not None and rule.target_key in by_source:
                visit(rule.target_key)
            visiting.remove(key)
            visited.add(key)

        for key in by_source:
            visit(key)

    def resolve(self, *, camera_id: str, source_id: int, raw_id: str,
                frame_id: int) -> dict[str, Any]:
        matches = [rule for rule in self._by_source_id.get((camera_id, source_id, str(raw_id)), [])
                   if rule.valid_from_frame <= frame_id <= rule.valid_until_frame]
        if len(matches) > 1:
            raise AdoptionAliasError("observation matches multiple adoption aliases")
        if not matches:
            return {"display_id": str(raw_id), "applied": False, "lineage": []}
        rule = matches[0]
        lineage: list[dict[str, Any]] = []
        visited: set[tuple[str, int, str, int]] = set()
        while True:
            if rule.source_key in visited:
                raise AdoptionAliasError("cycle detected while resolving adoption lineage")
            visited.add(rule.source_key)
            lineage.append({
                "event_type": "ADOPTED_ID",
                "camera_id": rule.camera_id,
                "source_id": rule.source_id,
                "raw_id": rule.raw_id,
                "display_id": rule.display_id,
                "adoption_frame": rule.adoption_frame,
                "event_timestamp_ns": rule.event_timestamp_ns,
            })
            next_rule = self._by_source_key.get(rule.target_key)
            if next_rule is None:
                break
            rule = next_rule
        return {"display_id": lineage[-1]["display_id"], "applied": True, "lineage": lineage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind verified MV3DT ADOPTED_ID records to offline review input hashes.")
    parser.add_argument("--alias-map", required=True, type=Path)
    parser.add_argument("--mqtt-decoded", required=True, type=Path)
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--mqtt-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_bound_manifest(args.alias_map, args.mqtt_decoded, args.event_log, args.mqtt_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_umask = os.umask(0o077)
    try:
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        os.umask(old_umask)
    os.chmod(args.output, 0o600)
    print(json.dumps({"output": str(args.output), "aliases": len(manifest["aliases"]),
                      "source_log_sha256": manifest["source_log_sha256"],
                      "mqtt_raw_sha256": manifest["mqtt_raw_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

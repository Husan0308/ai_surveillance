from __future__ import annotations

from services.frontend.app.realtime_models import (
    CameraMetadata,
    LatestMetadataStore,
    letterbox_rect,
    map_bbox_to_widget,
    parse_camera_rows,
    parse_track_message,
)


def test_parse_camera_rows_deduplicates_and_builds_stream_url() -> None:
    rows = parse_camera_rows(
        {
            "cameras": [
                {"id": "CAM-01", "online": True, "fps": "14.5", "width": 1920, "height": 1080},
                {"id": "CAM-01", "online": False},
                {"id": "CAM-02", "online": True, "stream_url": "/custom/cam02.mjpg"},
            ]
        }
    )

    assert [row.camera_id for row in rows] == ["CAM-01", "CAM-02"]
    assert rows[0].stream_url == "/api/v1/cameras/CAM-01/stream.mjpg"
    assert rows[0].fps == 14.5
    assert rows[1].stream_url == "/custom/cam02.mjpg"


def test_parse_camera_rows_survives_bad_numbers() -> None:
    rows = parse_camera_rows(
        {"cameras": [{"id": "CAM-01", "fps": "bad", "width": None, "height": "bad"}]}
    )
    assert len(rows) == 1
    assert rows[0].fps == 0.0
    assert rows[0].width == 1
    assert rows[0].height == 1


def test_parse_track_message_filters_invalid_tracks_and_clamps_confidence() -> None:
    parsed = parse_track_message(
        {
            "type": "tracks",
            "camera_id": "CAM-01",
            "frame_seq": "7",
            "timestamp_ns": "1000",
            "source_width": 1920,
            "source_height": 1080,
            "online": True,
            "fps": "14.2",
            "tracks": [
                {
                    "track_id": "CAM-01-T00007",
                    "state": "tracked",
                    "confidence": 1.4,
                    "bbox_xyxy": [100, 200, 500, 900],
                },
                {"track_id": "bad", "bbox_xyxy": [10, 10, 5, 5]},
                {"track_id": "bad2", "bbox_xyxy": ["x", 1, 2, 3]},
            ],
        }
    )

    assert parsed is not None
    camera_id, metadata = parsed
    assert camera_id == "CAM-01"
    assert metadata.frame_seq == 7
    assert metadata.timestamp_ns == 1000
    assert len(metadata.tracks) == 1
    assert metadata.tracks[0].confidence == 1.0


def test_latest_metadata_store_rejects_older_updates() -> None:
    store = LatestMetadataStore()
    newest = CameraMetadata(frame_seq=5, timestamp_ns=200)
    older_time = CameraMetadata(frame_seq=99, timestamp_ns=100)
    older_seq_same_time = CameraMetadata(frame_seq=4, timestamp_ns=200)

    assert store.update("CAM-01", newest) is True
    assert store.update("CAM-01", older_time) is False
    assert store.update("CAM-01", older_seq_same_time) is False
    assert store.get("CAM-01") is newest


def test_letterbox_bbox_mapping_matches_drawn_image() -> None:
    image_rect = letterbox_rect(0, 0, 1000, 1000, 1920, 1080)
    x, y, width, height, scale = image_rect

    assert round(x, 3) == 0.0
    assert round(width, 3) == 1000.0
    assert round(height, 3) == 562.5
    assert round(y, 3) == 218.75
    assert scale > 0

    mapped = map_bbox_to_widget((0, 0, 1920, 1080), image_rect)
    assert tuple(round(value, 3) for value in mapped) == (0.0, 218.75, 1000.0, 781.25)

import time
from pathlib import Path

import pytest

from services.camera_v11.source_ownership import SourceOwnership
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter
from scripts.run_preview_foundation import command


def test_duplicate_owner_rejected_before_stream_open(tmp_path):
    with SourceOwnership(["CAM-01", "CAM-04"], tmp_path):
        with pytest.raises(RuntimeError, match="another camera owner"):
            with SourceOwnership(["CAM-04"], tmp_path):
                pytest.fail("camera must not open")
        # Failed multi-camera acquisition must release the earlier CAM-02 lock.
        with pytest.raises(RuntimeError):
            with SourceOwnership(["CAM-02", "CAM-04"], tmp_path):
                pass
        with SourceOwnership(["CAM-02"], tmp_path):
            pass
    with SourceOwnership(["CAM-01", "CAM-04"], tmp_path):
        pass
    assert len(list(tmp_path.glob("*.lock"))) == 3  # Keep stable flock inodes.


@pytest.mark.parametrize("cameras", [[], ["CAM-01", "CAM-01"], ["../../escape"]])
def test_invalid_selection_fails_closed(cameras, tmp_path):
    with pytest.raises(ValueError):
        SourceOwnership(cameras, tmp_path)


def test_stale_and_future_frames_are_not_live(tmp_path):
    path = str(tmp_path / "preview")
    writer, reader = PreviewFrameWriter(path, 2, 2), PreviewFrameReader(path)
    try:
        assert reader.read_latest() is None  # Existing empty file isn't LIVE.
        writer.publish(bytes(16), timestamp_ns=time.monotonic_ns() - 2_000_000_000)
        assert reader.read_latest(max_age_sec=1) is None
        writer.publish(bytes(16), timestamp_ns=time.monotonic_ns() + 2_000_000_000)
        assert reader.read_latest() is None
        seq = writer.publish(bytes(16))
        assert reader.read_latest(metadata_only=True).payload == b""
        assert reader.read_latest().payload == bytes(16)
        assert reader.read_latest(after_sequence=seq) is None
    finally:
        reader.close()
        writer.close()


def test_restart_uses_new_inode_not_stale_mmap(tmp_path):
    path = str(tmp_path / "preview")
    reader = PreviewFrameReader(path)
    old = PreviewFrameWriter(path, 2, 2)
    old.publish(bytes([1]) * 16)
    old.publish(bytes([1]) * 16)
    assert reader.read_latest().sequence == 2
    old.close(unlink=False)
    new = PreviewFrameWriter(path, 2, 2)
    try:
        new.publish(bytes([2]) * 16)
        fresh = reader.read_latest(after_sequence=2)
        assert fresh.sequence == 1 and fresh.payload == bytes([2]) * 16
    finally:
        reader.close()
        new.close()


def test_container_uses_pinned_platform_and_readonly_f1_runtime(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    python = root / ".runtime/full-stack-venv/bin/python"
    env_file = tmp_path / "camera.env"
    env_file.write_text("# authorized environment fixture\n")
    monkeypatch.setenv("V11_ENV_FILE", str(env_file))
    cmd = command(root, python, tmp_path, "CAM-01,CAM-04", 30)
    assert "--rm" in cmd and "--init" in cmd
    assert "--pull=never" in cmd
    assert f"type=bind,src={root},dst={root},readonly" in cmd
    assert "V11_ENV_FILE=/run/camera.env" in cmd
    assert any(v.endswith("dst=/run/camera.env,readonly") for v in cmd)
    assert any(v.startswith("nvcr.io/nvidia/deepstream@sha256:") for v in cmd)
    assert str(python) in cmd and "services.camera_v11.preview_only_runtime" in cmd
    assert not any(v in cmd for v in ("--privileged", "nvinfer", "nvtracker", "pip", "sudo"))
    assert "Trash" not in " ".join(cmd)


@pytest.mark.parametrize("encoding", ["H264", "H265"])
def test_sdp_codec_discovery_does_not_need_rtp(encoding):
    from scripts.capture_dev_room_mv3dt import GstSdp, _codec_from_sdp
    _, sdp = GstSdp.SDPMessage.new()
    text = ("v=0\r\ns=fixture\r\n"
            "m=audio 0 RTP/AVP 8\r\na=rtpmap:8 PCMA/8000\r\n"
            f"m=video 0 RTP/AVP 96\r\na=rtpmap:96 {encoding}/90000\r\n")
    assert GstSdp.sdp_message_parse_buffer(text.encode(), sdp) == GstSdp.SDPResult.OK
    assert _codec_from_sdp(sdp) == encoding


def test_ambiguous_sdp_fails_closed():
    from scripts.capture_dev_room_mv3dt import GstSdp, _codec_from_sdp
    _, sdp = GstSdp.SDPMessage.new()
    text = ("v=0\r\ns=fixture\r\nm=video 0 RTP/AVP 96 97\r\n"
            "a=rtpmap:96 H264/90000\r\na=rtpmap:97 H265/90000\r\n")
    GstSdp.sdp_message_parse_buffer(text.encode(), sdp)
    with pytest.raises(ValueError, match="one supported"):
        _codec_from_sdp(sdp)

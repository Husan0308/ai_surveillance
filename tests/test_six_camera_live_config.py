"""CPU-only tests. No RTSP, Docker runtime or GPU inference."""
from pathlib import Path
from types import SimpleNamespace
import configparser
import json

import pytest

from scripts.six_camera_live.config import (
    Camera, CAMERAS, CHANNELS, build_app_config, camera_uri,
    validate_infer_config, validate_model_files,
)
from scripts.six_camera_live import launch


def cameras():
    return [
        Camera(cid, f"rtsp://nvr.example.invalid:554/Streaming/Channels/{channel}",
               "shared-reader", "synthetic-pass!")
        for cid, channel in zip(CAMERAS, CHANNELS)
    ]


def infer_config():
    return """
[property]
batch-size=6
network-mode=2
network-type=0
num-detected-classes=3
filter-out-class-ids=1;2
model-engine-file=/models/peopleNet/engines/known-b6.engine
onnx-file=/models/peopleNet/model/resnet34_peoplenet.onnx
labelfile-path=/models/peopleNet/config/labels.txt
"""


def test_six_source_order_and_three_by_two_wall():
    private, public = build_app_config(cameras())
    ini = configparser.ConfigParser(interpolation=None)
    ini.read_string(private)
    assert ini["tiled-display"]["rows"] == "2"
    assert ini["tiled-display"]["columns"] == "3"
    assert ini["streammux"]["live-source"] == "1"
    assert ini["streammux"]["batch-size"] == "6"
    assert ini["sink0"]["type"] == "2"
    assert ini["sink0"]["sync"] == "0"
    assert ini["sink0"]["qos"] == "0"
    assert ini["primary-gie"]["config-file"] == "/run/live/infer.txt"
    assert ini["tracker"]["tracker-width"] == "960"
    assert ini["tracker"]["tracker-height"] == "544"
    for index, channel in enumerate(CHANNELS):
        src = ini[f"source{index}"]
        assert src["type"] == "4"
        assert src["select-rtp-protocol"] == "4"
        assert src["uri"].endswith(f"/Streaming/Channels/{channel}")
        assert src["uri"].startswith("rtsp://shared-reader:synthetic-pass%21@")
    assert len(public["source_indices"]) == 6
    assert public["channels"] == list(CHANNELS)
    assert "synthetic-pass" not in json.dumps(public)


def test_invalid_source_order_fails():
    wrong = cameras()
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError, match="in order"):
        build_app_config(wrong)


@pytest.mark.parametrize("bad_uri", [
    "file:///tmp/a.mp4",
    "rtsp://nvr.example.invalid:554/Streaming/Channels/777",
    "rtsp://embedded:secret@nvr.example.invalid:554/Streaming/Channels/101",
    "rtsp://nvr.example.invalid:554/Streaming/Channels/101\nmalicious=1",
    "rtsp://nvr.example.invalid:abc/Streaming/Channels/101",
])
def test_unsafe_routes_rejected(bad_uri):
    with pytest.raises(ValueError):
        camera_uri(Camera("CAM-01", bad_uri, "test", "test"), 101)


def test_mixed_nvr_endpoints_rejected():
    items = cameras()
    items[2] = Camera("CAM-03", "rtsp://different.example.invalid/Streaming/Channels/301",
                      "reader", "secret")
    with pytest.raises(ValueError, match="one explicitly reviewed"):
        build_app_config(items)


def test_credentials_encoded_without_logging():
    uri, hostname, port = camera_uri(
        Camera("CAM-05", "rtsp://nvr.example.invalid/Streaming/Channels/501",
               "reader@team", "pass:word"), 501
    )
    assert "reader%40team:pass%3Aword@" in uri
    assert hostname == "nvr.example.invalid"
    assert port == 554


@pytest.mark.parametrize("replacement", [
    "batch-size=2",
    "network-mode=1",
    "network-type=1",
    "filter-out-class-ids=2",
    "model-engine-file=/tmp/engine",
])
def test_unvalidated_infer_rejected(replacement):
    text = infer_config()
    key = replacement.split("=")[0]
    original = next(line for line in text.splitlines() if line.startswith(key + "="))
    with pytest.raises(ValueError):
        validate_infer_config(text.replace(original, replacement))


def test_local_model_assets_are_validated(tmp_path):
    for path in (
        "peopleNet/engines/known-b6.engine",
        "peopleNet/model/resnet34_peoplenet.onnx",
        "peopleNet/config/labels.txt",
    ):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic file")
    got = validate_model_files(infer_config(), tmp_path)
    assert set(got) == {"model-engine-file", "onnx-file", "labelfile-path"}
    (tmp_path / "peopleNet/engines/known-b6.engine").unlink()
    with pytest.raises(FileNotFoundError):
        validate_model_files(infer_config(), tmp_path)


def test_secret_tar_not_exposed_in_docker_arguments(tmp_path):
    private, _ = build_app_config(cameras())
    packed = launch.secret_tar(private, infer_config())
    assert b"synthetic-pass" in packed
    argv = launch._docker_args("pinned-image", tmp_path, "restricted-camera-net",
                               ":0", tmp_path / "xauth", "owned-preview-test")
    cmd = " ".join(argv)
    assert "synthetic-pass" not in cmd
    assert "--network restricted-camera-net" in cmd
    assert "--log-driver=none" in cmd
    assert "--read-only" in argv
    assert "tar -xf - -C /run/live" in cmd
    assert "--tiledtext" in cmd


def test_forbidden_unrestricted_networks_rejected():
    for network in ("", "host", "bridge", "default", "none"):
        with pytest.raises(ValueError):
            launch.checked_network(network, "nvr.example.invalid", 554)


def test_check_mode_never_starts_docker(monkeypatch, tmp_path, capsys):
    from scripts.six_camera_live import launch as module

    model_root = tmp_path / "models"
    for rel in (
        "peopleNet/engines/known-b6.engine",
        "peopleNet/model/resnet34_peoplenet.onnx",
        "peopleNet/config/labels.txt",
    ):
        p = model_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic")
    infer = tmp_path / "infer.txt"
    infer.write_text(infer_config())
    platform = tmp_path / "platform.json"
    platform.write_text(json.dumps({"image_tag": "deepstream-9.1-test",
                                    "image": "sha256:synthetic"}))
    monkeypatch.setattr(module, "PLATFORM", platform)
    monkeypatch.setattr(module, "load_settings",
                        lambda unused: SimpleNamespace(cameras=[
                            SimpleNamespace(camera_id=x.camera_id, uri=x.uri,
                                            username=x.username, password=x.password)
                            for x in cameras()
                        ]))
    def forbid_any_external(*args, **kwargs):
        raise AssertionError("check-mode must not run any external program")
    monkeypatch.setattr(module.subprocess, "run", forbid_any_external)
    monkeypatch.setattr(module.subprocess, "Popen", forbid_any_external)
    result = module.main(["--check", "--infer-config", str(infer),
                          "--models-root", str(model_root)])
    assert result == 0
    captured = capsys.readouterr().out
    assert "CONFIG PREPARATION: PASS" in captured
    assert "synthetic-pass" not in captured
    assert "nvr.example.invalid" not in captured


def test_run_needs_three_explicit_approvals(monkeypatch, tmp_path):
    from scripts.six_camera_live import launch as module
    model_root = tmp_path / "models"
    for rel in (
        "peopleNet/engines/known-b6.engine",
        "peopleNet/model/resnet34_peoplenet.onnx",
        "peopleNet/config/labels.txt",
    ):
        p = model_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    infer = tmp_path / "infer.txt"
    infer.write_text(infer_config())
    platform = tmp_path / "platform.json"
    platform.write_text(json.dumps({"image_tag": "deepstream-9.1-test",
                                    "image": "sha256:synthetic"}))
    monkeypatch.setattr(module, "PLATFORM", platform)
    monkeypatch.setattr(module, "load_settings",
                        lambda unused: SimpleNamespace(cameras=[
                            SimpleNamespace(camera_id=x.camera_id, uri=x.uri,
                                            username=x.username, password=x.password)
                            for x in cameras()
                        ]))
    monkeypatch.setattr(module, "display_preflight",
                        lambda: (_ for _ in ()).throw(AssertionError("approval not reached")))
    assert module.main(["--run", "--infer-config", str(infer),
                        "--models-root", str(model_root)]) == 2
    # Even explicit authorization is not enough without the pinned engine hash.
    assert module.main(["--run", "--infer-config", str(infer),
                        "--models-root", str(model_root),
                        "--authorize-physical-rtsp",
                        "--nvr-readonly-authorized",
                        "--network-egress-verified"]) == 2

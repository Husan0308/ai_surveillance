import pytest

from services.camera_v11.preview_only_runtime import DEFAULT_CAMERAS, main, preview_camera_ids


def test_preview_only_defaults_do_not_open_room_pair():
    assert preview_camera_ids(",".join(DEFAULT_CAMERAS)) == ("CAM-02", "CAM-03", "CAM-05", "CAM-06")


@pytest.mark.parametrize("selection", ["CAM-01", "CAM-04", "CAM-01,CAM-02", "CAM-02,CAM-04",
                                        "CAM-02,CAM-02", "", ",,"])
def test_invalid_ownership_or_duplicate_camera_is_rejected(selection):
    with pytest.raises(ValueError):
        preview_camera_ids(selection)


def test_invalid_ownership_exits_before_loading_credentials_or_starting_sources(monkeypatch):
    import services.camera_v11.preview_only_runtime as preview

    monkeypatch.setattr("sys.argv", ["preview_only_runtime", "--cameras", "CAM-01"])
    monkeypatch.setattr(preview, "load_settings", lambda: pytest.fail("loaded settings before ownership guard"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2

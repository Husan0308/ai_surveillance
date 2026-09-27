from __future__ import annotations

from services.ml_service.app.deepstream.decoder_sample import BoundedDecoderSample


def test_decoder_sample_is_bounded_and_flushes(tmp_path) -> None:
    sample = BoundedDecoderSample(tmp_path, "CAM-04", "h265", max_bytes=64)

    assert sample.append(bytes(range(50))) == 50
    assert sample.append(bytes(range(50))) == 14
    assert sample.append(b"ignored") == 0
    sample.close()

    payload = (tmp_path / "cam_04.h265").read_bytes()
    assert len(payload) == 64
    assert payload == bytes(range(50)) + bytes(range(14))


def test_decoder_sample_validates_codec(tmp_path) -> None:
    try:
        BoundedDecoderSample(tmp_path, "CAM-01", "av1")
    except ValueError as exc:
        assert "unsupported decoder sample codec" in str(exc)
    else:
        raise AssertionError("unsupported codec must fail explicitly")

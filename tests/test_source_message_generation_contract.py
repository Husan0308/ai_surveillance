from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_retired_source_messages_are_separated_from_shared_fatal_errors():
    main = (NATIVE / "main.cpp").read_text(encoding="utf-8")
    helper = (NATIVE / "source_message_owner.hpp").read_text(encoding="utf-8")

    assert '#include "source_message_owner.hpp"' in main
    assert "source_message_owner::attach(GST_OBJECT(e), ctx, owner_tag.generation)" in main
    assert "source_message_owner::attach(GST_OBJECT(source), ctx, generation)" in main
    assert "STALE_SOURCE_GENERATION_MESSAGE" in main
    assert "owner.stale" in main
    assert "source_message_owner::resolve(" in main
    assert "if (error && !ctx) fail(\"shared mux/output error: \" + message);" in main
    assert "enum class Disposition { CURRENT_SOURCE, STALE_SOURCE, SHARED_OR_UNKNOWN };" in helper
    assert "tag->generation != current_generation" in helper
    assert "inline Disposition resolve(" in helper

from __future__ import annotations

import numpy as np
import pytest

from services.mv3dt_staging.person_identity_memory import (
    AdoptionEvent,
    IdentityMemoryError,
    IdentityObservation,
    MemoryConfig,
    PersonIdentityMemory,
)


def obs(raw: str, time_s: float, vector=(1.0, 0.0), *, cam="CAM-02", quality=0.95,
        xy=(0.0, 0.0), validated=True, scope="camera_local") -> IdentityObservation:
    return IdentityObservation(
        run_id="run-test", camera_id=cam, raw_id=raw, timestamp_seconds=time_s,
        embedding=np.asarray(vector, dtype=np.float32), embedding_quality=quality,
        world_xy=xy, world_geometry_validated=validated, raw_id_scope=scope,
    )


def config(**kwargs) -> MemoryConfig:
    values = {
        "retention_seconds": 30,
        "active_timeout_seconds": 1,
        "min_embedding_quality": 0.75,
        "min_similarity": 0.90,
        "min_best_second_margin": 0.10,
        "threshold_profile_id": "synthetic-fixture-only",
        "max_speed_units_per_second": 10,
        "allowed_camera_transitions": frozenset({frozenset({"CAM-02", "CAM-05"})}),
    }
    values.update(kwargs)
    return MemoryConfig(**values)


def test_raw_and_persistent_ids_are_separate_and_default_is_fail_closed() -> None:
    memory = PersonIdentityMemory(MemoryConfig(), run_id="run-test")
    first = memory.observe(obs("7", 0.0))
    returned = memory.observe(obs("8", 5.0))

    assert first["raw_id"] == "7"
    assert first["persistent_person_id"].startswith("person-provisional-")
    assert first["persistent_person_id"] != "7"
    assert returned["raw_id"] == "8"
    assert returned["persistent_person_id"] != first["persistent_person_id"]
    assert returned["match_status"] == "provisional_uncalibrated_thresholds"
    assert returned["match_confidence"] == pytest.approx(1.0)
    assert returned["identity_is_recognized"] is False


def test_calibrated_clear_match_reuses_person_but_preserves_new_raw_id() -> None:
    memory = PersonIdentityMemory(config(), run_id="run-test")
    first = memory.observe(obs("7", 0.0, xy=(0, 0)))
    second = memory.observe(obs("12", 6.0, xy=(10, 0)))

    assert second["raw_id"] == "12"
    assert second["persistent_person_id"] == first["persistent_person_id"]
    assert second["match_status"] == "appearance_reidentified"
    assert second["match_confidence"] == pytest.approx(1.0)
    assert second["display_id"] == second["persistent_person_id"]


def test_first_usable_embedding_can_reidentify_after_raw_track_was_seen() -> None:
    memory = PersonIdentityMemory(config(), run_id="run-test")
    original = memory.observe(obs("7", 0.0, xy=(0, 0)))
    new_track_without_feature = memory.observe(
        IdentityObservation(
            run_id="run-test", camera_id="CAM-02", raw_id="12", timestamp_seconds=5.0,
            embedding=None, embedding_quality=0.0, world_xy=(10, 0),
            world_geometry_validated=True,
        )
    )
    assert new_track_without_feature["persistent_person_id"] != original["persistent_person_id"]

    first_good_feature = memory.observe(obs("12", 5.1, xy=(10, 0)))
    assert first_good_feature["raw_id"] == "12"
    assert first_good_feature["persistent_person_id"] == original["persistent_person_id"]
    assert first_good_feature["match_status"] == "appearance_reidentified"
    assert memory.snapshot()["memory_size"] == 1


def test_close_second_candidate_stays_provisional() -> None:
    memory = PersonIdentityMemory(config(max_speed_units_per_second=1000), run_id="run-test")
    memory.observe(obs("a", 0, (1, 0), xy=(0, 0)))
    memory.observe(obs("b", 0.1, (0.995, 0.1), xy=(20, 0)))
    result = memory.observe(obs("c", 5, (1, 0), xy=(0, 0)))

    assert result["match_status"] == "provisional_ambiguous_margin"
    assert len(result["candidate_scores"]) == 2
    assert result["persistent_person_id"] not in {
        row["persistent_person_id"] for row in memory._last_decisions[:2]
    }


def test_different_simultaneously_active_raw_track_cannot_share_person_id() -> None:
    memory = PersonIdentityMemory(config(active_timeout_seconds=2), run_id="run-test")
    first = memory.observe(obs("7", 0, (1, 0)))
    memory.observe(obs("7", 1, (1, 0)))
    second = memory.observe(obs("8", 1.2, (1, 0)))

    assert second["persistent_person_id"] != first["persistent_person_id"]
    assert second["match_status"] == "provisional_no_eligible_gallery_match"


def test_low_quality_embedding_is_not_added_or_matched() -> None:
    memory = PersonIdentityMemory(config(), run_id="run-test")
    first = memory.observe(obs("7", 0, quality=0.4))
    second = memory.observe(obs("8", 5, quality=0.4))

    assert first["match_status"] == "provisional_embedding_unavailable_or_low_quality"
    assert second["candidate_scores"] == []
    assert memory.snapshot()["embedding_count"] == 0


def test_unvalidated_geometry_and_implausible_movement_block_match() -> None:
    memory = PersonIdentityMemory(config(), run_id="run-test")
    first = memory.observe(obs("7", 0, xy=(0, 0), validated=False))
    unvalidated = memory.observe(obs("8", 5, xy=(1, 0), validated=False))
    assert unvalidated["persistent_person_id"] != first["persistent_person_id"]
    assert unvalidated["match_status"] == "provisional_world_geometry_unvalidated"

    memory = PersonIdentityMemory(config(max_speed_units_per_second=1), run_id="run-test")
    first = memory.observe(obs("7", 0, xy=(0, 0)))
    moved = memory.observe(obs("8", 5, xy=(100, 0)))
    assert moved["persistent_person_id"] != first["persistent_person_id"]
    assert moved["match_status"] == "provisional_movement_implausible"


def test_expired_gallery_is_removed_and_old_raw_id_is_not_reused() -> None:
    memory = PersonIdentityMemory(config(retention_seconds=5), run_id="run-test")
    first = memory.observe(obs("7", 0))
    assert memory.expire(5.1) == [first["persistent_person_id"]]
    second = memory.observe(obs("8", 6))
    assert second["persistent_person_id"] != first["persistent_person_id"]
    assert memory.snapshot()["embedding_count"] == 1


def test_mv3dt_global_raw_id_is_shared_across_cameras_but_local_numeric_ids_are_not() -> None:
    memory = PersonIdentityMemory(run_id="run-test")
    global_cam02 = memory.observe(obs("7", 0, scope="mv3dt_global", cam="CAM-02"))
    global_cam05 = memory.observe(obs("7", 0.1, scope="mv3dt_global", cam="CAM-05"))
    assert global_cam05["persistent_person_id"] == global_cam02["persistent_person_id"]

    local_a = memory.observe(obs("7", 1, scope="camera_local", cam="CAM-02"))
    local_b = memory.observe(obs("7", 1, scope="camera_local", cam="CAM-05"))
    assert local_a["persistent_person_id"] != local_b["persistent_person_id"]


def test_hash_bound_adoption_changes_display_only_not_person_identity() -> None:
    memory = PersonIdentityMemory(run_id="run-test")
    old = memory.observe(obs("1", 0))
    new = memory.observe(obs("2", 4, (0, 1)))
    event = AdoptionEvent("abc123", "CAM-02", 0, "1", "2", 6)

    with pytest.raises(IdentityMemoryError, match="hash mismatch"):
        memory.apply_confirmed_adoption(event, expected_source_log_sha256="not-the-source")
    result = memory.apply_confirmed_adoption(event, expected_source_log_sha256="abc123")
    old_after = memory.observe(obs("1", 4.1))
    new_after = memory.observe(obs("2", 4.2, (0, 1)))

    assert result["persistent_ids_merged"] is False
    assert old_after["persistent_person_id"] == old["persistent_person_id"]
    assert new_after["persistent_person_id"] == new["persistent_person_id"]
    assert old_after["persistent_person_id"] != new_after["persistent_person_id"]
    assert old_after["display_id"] == new_after["display_id"]


def test_adoption_alias_cycle_and_live_retroactive_use_are_rejected() -> None:
    memory = PersonIdentityMemory(run_id="run-test")
    first = AdoptionEvent("abc", "CAM-02", 0, "1", "2", 6)
    second = AdoptionEvent("abc", "CAM-02", 0, "2", "1", 7)
    memory.apply_confirmed_adoption(first, expected_source_log_sha256="abc")
    with pytest.raises(IdentityMemoryError, match="cycle"):
        memory.apply_confirmed_adoption(second, expected_source_log_sha256="abc")
    with pytest.raises(IdentityMemoryError, match="offline replay only"):
        memory.apply_confirmed_adoption(first, expected_source_log_sha256="abc", retrospective_offline=False)


def test_out_of_order_observation_is_rejected() -> None:
    memory = PersonIdentityMemory(run_id="run-test")
    memory.observe(obs("7", 2.0))
    with pytest.raises(IdentityMemoryError, match="out of order"):
        memory.observe(obs("7", 1.0))


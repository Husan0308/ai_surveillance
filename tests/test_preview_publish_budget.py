from services.camera_v11.preview_only_runtime import PreviewPublishBudget


def test_20fps_with_arrival_jitter_is_not_throttled():
    budget = PreviewPublishBudget(20, 0.0)
    arrivals = [i * 0.05 + (0.02 if i % 2 else 0.0) for i in range(600)]
    assert all(budget.allow(now) for now in arrivals)


def test_faster_source_still_has_bounded_write_rate():
    budget = PreviewPublishBudget(20, 0.0)
    accepted = sum(budget.allow(i / 60) for i in range(601))
    assert 200 <= accepted <= 202


def test_long_pause_does_not_accumulate_a_write_burst():
    budget = PreviewPublishBudget(20, 0.0)
    assert budget.allow(0.0)
    assert sum(budget.allow(3600.0) for _ in range(100)) == 2


def test_source_timeline_reset_releases_old_rate_limit():
    budget = PreviewPublishBudget(20, 10.0)
    assert budget.allow(10.0)
    assert budget.allow(10.0)
    assert budget.allow(0.0)
    assert budget.allow(0.05)

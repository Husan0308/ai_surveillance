from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / '.runtime/deepstream-fix5-generation-20261031/sources/apps/sample_apps/deepstream-app/six_live_monitor.c'
PREVIEW = Path(__file__).resolve().parents[1] / '.runtime/deepstream-fix5-generation-20261031/sources/apps/sample_apps/deepstream-app/mosaic_preview_branch.c'
REPRODUCER = Path(__file__).resolve().parents[1] / 'tests/native/fix5_inflight_reproducer.c'
HARNESS = Path(__file__).resolve().parents[1] / 'tests/native/fix5_generation_harness.c'


def function(source, start, end):
    return source.split(start, 1)[1].split(end, 1)[0]


def test_contended_generation_snapshot_is_skipped():
    source = SOURCE.read_text()
    tick = function(source, 'static gboolean tick (gpointer data)', 'static gboolean attach_probe')
    assert tick.index('g_mutex_lock (&monitor_lock)') < tick.index('g_mutex_trylock (&telemetry_lock)')
    busy = tick.split('if (!g_mutex_trylock (&telemetry_lock))', 1)[1].split('memcpy (generations', 1)[0]
    assert 'g_mutex_unlock (&monitor_lock)' in busy
    assert 'return G_SOURCE_CONTINUE' in busy
    assert 'memset (generations' not in tick


def test_reconnect_quarantines_until_confirmed_null_boundary():
    source = SOURCE.read_text()
    reconnect = function(source, 'void six_live_source_reconnecting (guint id)', '/* Called only after reset_source_pipeline()')
    assert reconnect.index('g_mutex_lock (&monitor_lock)') < reconnect.index('g_mutex_lock (&telemetry_lock)')
    assert 'source_epoch_ready[id] = FALSE' in reconnect
    assert 'source_generation[id]++' not in reconnect

    boundary = function(source, 'void six_live_source_generation_boundary (guint id)', 'static gpointer source_owner_copy')
    assert boundary.index('source_generation[id]++') < boundary.index('memset (&stages[stage][id]')
    assert 'last_counts[id] = 0' in boundary
    assert 'source_epoch_ready[id] = TRUE' in boundary


def test_frame_generation_is_attached_before_mux_and_required_at_every_stage():
    source = SOURCE.read_text()
    attach = function(source, 'gboolean six_live_attach (AppCtx *app)', 'void six_live_stop (void)')
    assert 'gst_element_get_static_pad (source->bin, "src")' in attach
    assert 'source_lineage_probe' in attach
    assert 'generation_begin_hook' in attach and 'generation_boundary_hook' in attach

    ownership = function(source, 'static SourceOwnerResult source_frame_owner', 'gboolean six_live_preview_frame_generation')
    assert 'frame_user_meta_list' in ownership
    assert 'source_generation[fm->source_id] == token->generation' in ownership

    probe = function(source, 'static GstPadProbeReturn metadata_probe', 'static GstPadProbeReturn labels_probe')
    assert probe.index('source_frame_owner (fm, &owner)') < probe.index('update(stage,fm->source_id')
    assert 'snapshot_tracker_frame (fm, buffer, &owner)' in probe


def test_preview_uses_exact_frame_owner_token_not_reusable_frame_pts_keys():
    source = SOURCE.read_text()
    preview = PREVIEW.read_text()
    owner = function(source, 'gboolean six_live_preview_frame_generation', 'void six_live_preview_frame')
    assert 'source_frame_owner (frame_meta, &owner)' in owner
    assert 'owner.generation' in owner
    remember = function(preview, 'remember_mosaic_frame_identity (', 'static void\nnote_published_mosaic_sources')
    assert 'six_live_preview_frame_generation (meta, &source->source_generation)' in remember
    assert 'six_live_preview_identity_generation' not in preview
    assert 'source_frame_identities' not in SOURCE.read_text()
    harness = (Path(__file__).resolve().parents[1] / 'tests/native/fix5_generation_harness.c').read_text()
    assert 'assert_preview_owner(&frame0, FALSE, 0)' in harness
    assert 'assert_preview_owner(&frame1, TRUE, 1)' in harness


def test_source_epoch_advances_only_after_source_bin_null_is_confirmed():
    sources = (SOURCE.parents[3] / 'apps/apps-common/src/deepstream_source_bin.c').read_text()
    reset = function(sources, 'reset_source_pipeline (gpointer data)', '/* Function called at regular interval')
    null_request = reset.index('gst_element_set_state (src_bin->bin, GST_STATE_NULL)')
    null_confirm = reset.index('gst_element_get_state (src_bin->bin')
    boundary = reset.index('generation_boundary_hook (src_bin->bin_id)')
    restart = reset.index('gst_element_sync_state_with_parent (src_bin->bin)')
    assert null_request < null_confirm < boundary < restart
    assert 'state != GST_STATE_NULL' in reset
    assert 'pending != GST_STATE_VOID_PENDING' in reset


def test_original_inflight_reproducer_now_requires_stale_rejection():
    reproducer = REPRODUCER.read_text()
    harness = HARNESS.read_text()
    assert '#include "fix5_generation_harness.c"' in reproducer
    assert 'assert_rejected_all_stages(&g0, source_id)' in harness
    assert 'assert_rejected_all_stages(&g1, source_id)' in harness
    assert 'for (guint source_id = 0; source_id < 6; ++source_id)' in harness
    assert 'REPRODUCED: held generation-0 frame stamped as generation 1' not in reproducer

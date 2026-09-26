#include "bbox_history_lifecycle.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define CAPACITY 128
#define RETIRE_GAP 2

static int
create_history (Pn263HistoryPool *pool, Pn263HistorySlot *slots,
    uint8_t *claimed, uint64_t frame)
{
  int index = pn263_history_pool_acquire (pool, slots, CAPACITY, claimed, frame);
  assert (index >= 0);
  claimed[index] = 1;
  return index;
}

static void
test_many_sequential_native_tracks (void)
{
  Pn263HistoryPool pool;
  Pn263HistorySlot slots[CAPACITY] = { 0 };
  uint8_t claimed[CAPACITY];
  uint64_t frame;

  pn263_history_pool_init (&pool);
  for (frame = 0; frame < 50000; frame++) {
    memset (claimed, 0, sizeof (claimed));
    pn263_history_pool_begin_frame (&pool, slots, CAPACITY, frame, RETIRE_GAP);
    create_history (&pool, slots, claimed, frame);
    assert (pool.active_count <= 3);
    assert (pool.active_count <= CAPACITY);
  }
  assert (pool.created_count == 50000);
  assert (pool.reused_count > 0);
  assert (pool.high_water <= 3);
  puts ("PASS many sequential native tracks");
}

static void
test_fragmentation_and_track_id_reuse (void)
{
  Pn263HistoryPool pool;
  Pn263HistorySlot slots[CAPACITY] = { 0 };
  uint8_t claimed[CAPACITY] = { 0 };
  const int reused_native_track_id = 77;
  int index;
  uint64_t first_generation;

  pn263_history_pool_init (&pool);
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 10, RETIRE_GAP);
  index = create_history (&pool, slots, claimed, 10);
  first_generation = slots[index].generation;
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 11, RETIRE_GAP);
  assert (slots[index].used);             /* one-frame miss */
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 12, RETIRE_GAP);
  assert (slots[index].used);             /* two-frame tolerance */
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 13, RETIRE_GAP);
  assert (!slots[index].used);            /* third-frame gap retires */
  /* This pre-tracker history cannot see NvDCF IDs; every new temporal segment
   * receives a fresh generation even when a tracker later reuses its ID. */
  memset (claimed, 0, sizeof (claimed));
  index = create_history (&pool, slots, claimed, 14);
  assert (reused_native_track_id == 77);
  assert (slots[index].generation > first_generation);
  assert (pool.active_count == 1);
  puts ("PASS fragmentation and track ID reuse");
}

static void
test_disappearance_reentry_and_long_empty_camera (void)
{
  Pn263HistoryPool pool;
  Pn263HistorySlot slots[CAPACITY] = { 0 };
  uint8_t claimed[CAPACITY] = { 0 };
  int index;
  uint64_t generation;
  uint64_t frame;

  pn263_history_pool_init (&pool);
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 1, RETIRE_GAP);
  index = create_history (&pool, slots, claimed, 1);
  generation = slots[index].generation;
  for (frame = 2; frame <= 720001; frame++)
    pn263_history_pool_begin_frame (&pool, slots, CAPACITY, frame, RETIRE_GAP);
  assert (pool.active_count == 0);
  memset (claimed, 0, sizeof (claimed));
  pn263_history_pool_begin_frame (&pool, slots, CAPACITY, 720002, RETIRE_GAP);
  index = create_history (&pool, slots, claimed, 720002);
  assert (slots[index].generation > generation);
  puts ("PASS disappearance, re-entry, and long empty-camera retirement");
}

static void
test_thirty_minute_logical_operation_and_bounded_reuse (void)
{
  Pn263HistoryPool pool;
  Pn263HistorySlot slots[CAPACITY] = { 0 };
  uint8_t claimed[CAPACITY];
  uint64_t frame;

  pn263_history_pool_init (&pool);
  for (frame = 0; frame < 36000; frame++) { /* 30 min at 20 FPS */
    memset (claimed, 0, sizeof (claimed));
    pn263_history_pool_begin_frame (&pool, slots, CAPACITY, frame, RETIRE_GAP);
    if (frame % 7 == 0)
      create_history (&pool, slots, claimed, frame);
    assert (pool.active_count <= CAPACITY);
    assert (pool.high_water <= CAPACITY);
  }
  assert (pool.created_count > CAPACITY);
  assert (pool.reused_count > 0);
  assert (pn263_history_pool_clear (&pool, slots, CAPACITY) == 0);
  assert (pool.active_count == 0);
  puts ("PASS 30-minute logical operation and bounded reuse");
}

static void
test_frame_number_reset_and_protected_eviction (void)
{
  Pn263HistoryPool pool;
  Pn263HistorySlot slots[2] = { 0 };
  uint8_t claimed[2] = { 0 };
  int a, b, c;

  pn263_history_pool_init (&pool);
  pn263_history_pool_begin_frame (&pool, slots, 2, 100, RETIRE_GAP);
  a = pn263_history_pool_acquire (&pool, slots, 2, claimed, 100);
  claimed[a] = 1;
  b = pn263_history_pool_acquire (&pool, slots, 2, claimed, 100);
  claimed[b] = 1;
  assert (pool.active_count == 2);
  c = pn263_history_pool_acquire (&pool, slots, 2, claimed, 100);
  assert (c == -1);                       /* claimed slots cannot be evicted */
  memset (claimed, 0, sizeof (claimed));
  claimed[a] = 1;
  c = pn263_history_pool_acquire (&pool, slots, 2, claimed, 101);
  assert (c == b);                        /* oldest unclaimed slot is recycled */
  assert (slots[a].used);
  assert (pool.active_count == 2);
  assert (pool.evicted_count == 1);
  assert (pn263_history_pool_begin_frame (&pool, slots, 2, 0, RETIRE_GAP));
  assert (pool.active_count == 0);        /* reconnect frame reset */
  puts ("PASS frame reset and claimed-slot protection");
}

int
main (void)
{
  test_many_sequential_native_tracks ();
  test_fragmentation_and_track_id_reuse ();
  test_disappearance_reentry_and_long_empty_camera ();
  test_thirty_minute_logical_operation_and_bounded_reuse ();
  test_frame_number_reset_and_protected_eviction ();
  puts ("ALL bbox history lifecycle tests PASS");
  return 0;
}

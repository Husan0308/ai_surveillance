#include "bbox_history_lifecycle.h"

#include <limits.h>
#include <string.h>

void
pn263_history_pool_init (Pn263HistoryPool *pool)
{
  if (pool) memset (pool, 0, sizeof (*pool));
}

static void
retire_slot (Pn263HistoryPool *pool, Pn263HistorySlot *slot)
{
  if (!slot->used) return;
  slot->used = false;
  slot->last_frame = 0;
  if (pool->active_count > 0) pool->active_count--;
  pool->retired_count++;
}

bool
pn263_history_pool_begin_frame (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity, uint64_t frame,
    uint64_t max_gap)
{
  bool reset = false;
  size_t i;
  if (!pool || !slots) return false;

  if (pool->has_last_frame && frame < pool->last_frame) {
    for (i = 0; i < capacity; i++) retire_slot (pool, &slots[i]);
    pool->last_frame = frame;
    pool->has_last_frame = true;
    reset = true;
  } else {
    for (i = 0; i < capacity; i++) {
      Pn263HistorySlot *slot = &slots[i];
      if (slot->used && frame >= slot->last_frame &&
          frame - slot->last_frame > max_gap)
        retire_slot (pool, slot);
    }
    pool->last_frame = frame;
    pool->has_last_frame = true;
  }
  return reset;
}

int
pn263_history_pool_acquire (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity, const uint8_t *claimed,
    uint64_t frame)
{
  size_t i, chosen = capacity;
  uint64_t oldest = UINT64_MAX;
  bool had_generation = false;

  if (!pool || !slots || capacity == 0 || capacity > INT_MAX) return -1;
  for (i = 0; i < capacity; i++) {
    if (!slots[i].used) {
      chosen = i;
      break;
    }
  }
  if (chosen == capacity) {
    for (i = 0; i < capacity; i++) {
      if (claimed && claimed[i]) continue;
      if (slots[i].last_frame < oldest) {
        oldest = slots[i].last_frame;
        chosen = i;
      }
    }
    if (chosen == capacity) return -1;
    retire_slot (pool, &slots[chosen]);
    pool->evicted_count++;
    had_generation = slots[chosen].generation != 0;
  } else {
    had_generation = slots[chosen].generation != 0;
  }

  slots[chosen].used = true;
  slots[chosen].last_frame = frame;
  slots[chosen].generation++;
  if (slots[chosen].generation == 0) slots[chosen].generation = 1;
  pool->active_count++;
  pool->created_count++;
  if (had_generation) pool->reused_count++;
  if (pool->active_count > pool->high_water)
    pool->high_water = pool->active_count;
  return (int) chosen;
}

void
pn263_history_pool_touch (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t index, uint64_t frame)
{
  if (!pool || !slots || !slots[index].used) return;
  slots[index].last_frame = frame;
}

size_t
pn263_history_pool_clear (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity)
{
  size_t i;
  if (!pool || !slots) return 0;
  for (i = 0; i < capacity; i++) retire_slot (pool, &slots[i]);
  pool->has_last_frame = false;
  return pool->active_count;
}

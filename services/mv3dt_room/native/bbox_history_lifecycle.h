#ifndef PN263_BBOX_HISTORY_LIFECYCLE_H
#define PN263_BBOX_HISTORY_LIFECYCLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  bool used;
  uint64_t last_frame;
  uint64_t generation;
} Pn263HistorySlot;

typedef struct {
  size_t active_count;
  size_t high_water;
  uint64_t created_count;
  uint64_t retired_count;
  uint64_t evicted_count;
  uint64_t reused_count;
  uint64_t last_frame;
  bool has_last_frame;
} Pn263HistoryPool;

void pn263_history_pool_init (Pn263HistoryPool *pool);

/* Retire entries older than max_gap. A frame-number regression starts a new
 * source epoch and clears every slot so reconnects/resets cannot inherit old
 * temporal state. Returns true when a source epoch reset occurred. */
bool pn263_history_pool_begin_frame (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity, uint64_t frame,
    uint64_t max_gap);

/* Allocate a fresh generation in a free slot, or recycle the oldest slot not
 * claimed by the current frame. Returns -1 only if all slots are claimed. */
int pn263_history_pool_acquire (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity, const uint8_t *claimed,
    uint64_t frame);

void pn263_history_pool_touch (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t index, uint64_t frame);

/* Pipeline teardown retires all active histories and returns the final count. */
size_t pn263_history_pool_clear (Pn263HistoryPool *pool,
    Pn263HistorySlot *slots, size_t capacity);

#endif

# Camera Service Step 2

This branch exports a sparse, latest-only 672x378 BGR frame for each camera through `/dev/shm` while keeping the display path independent.

The camera service contains no detector, tracker, ReID, identity, API, or frontend code. A dead or slow ML consumer cannot apply backpressure because the ML tap uses a leaky one-buffer queue and drops frames before conversion when the configured tap interval has not elapsed.

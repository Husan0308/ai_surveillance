# V9.3 current-display bbox

V9.2 removed fabricated all-source empty updates and stale detector geometry. Its live walk test still showed overlay age around 65-68 ms median and about 189 ms p95 while the tracker ran at 7.4-8.0 Hz.

V9.3 keeps the V9.1 same-process TensorRT 8.6 primary-context architecture and the V9.2 temporal fixes. It reallocates a small amount of detector GPU duty to NvDCF (10 Hz target, detector budget 0.26) and adds a display-only, center-only, bounded extrapolation from the latest two real NvDCF boxes. The extrapolated rectangle is never fed back into detector association, NvDCF, identity state, or the persistent track cache. Size always comes from the latest real NvDCF box.

def format_metrics(metrics: dict) -> str:
    lines = []
    for camera_id, item in metrics["cameras"].items():
        if item["is_starved"]:
            lines.append(f"{camera_id} STARVED frame:{item['recv_frame_id']} starved:{item['starved_count']}")
        else:
            lines.append(f"{camera_id} src:{item['source_fps']:.1f} fps frame:{item['recv_frame_id']} used:{item['used_frame_id']} age:{item['frame_age_ms']:.0f}ms interarrival:{item['interarrival_ms']:.1f}ms max:{item['max_interarrival_ms']:.1f}ms drops:{item['dropped_old']} dup:{item['duplicate_count']}")
    lines.append(f"BATCH size:{metrics['batch_size']} cameras:{metrics['batch_cameras']} batch_rate:{metrics['batch_rate']:.1f}/s stale_drops:{metrics['stale_drops']}")
    lines.append(f"FLOW source_total_fps:{metrics['source_total_fps']:.1f} processed_total_fps:{metrics['processed_total_fps']:.1f} dropped_total_fps:{metrics['dropped_total_fps']:.1f}")
    return "\n".join(lines)

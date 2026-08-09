def format_metrics(metrics: dict) -> str:
    lines = []
    for camera_id, item in metrics["cameras"].items():
        if item["is_starved"]:
            lines.append(f"{camera_id} STARVED backend:{item.get('backend','offline')} frame:{item['recv_frame_id']} starved:{item['starved_count']}")
        else:
            lines.append(f"{camera_id} {item.get('backend','offline')} src:{item['source_fps']:.1f} fps {item.get('width',0)}x{item.get('height',0)} frame:{item['recv_frame_id']} used:{item['used_frame_id']} age:{item['frame_age_ms']:.0f}ms mean:{item.get('frame_mean',0):.1f} var:{item.get('frame_variance',0):.1f} detector_inputs:{item.get('detector_inputs',0)} detections:{item.get('detections',0)} drops:{item['dropped_old']} dup:{item['duplicate_count']}")
    lines.append(f"BATCH size:{metrics['batch_size']} cameras:{metrics['batch_cameras']} batch_rate:{metrics['batch_rate']:.1f}/s stale_drops:{metrics['stale_drops']}")
    lines.append(f"FLOW source_total_fps:{metrics['source_total_fps']:.1f} processed_total_fps:{metrics['processed_total_fps']:.1f} dropped_total_fps:{metrics['dropped_total_fps']:.1f}")
    return "\n".join(lines)

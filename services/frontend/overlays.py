def detection_overlay(detection: dict) -> dict:
    """Convert API detection metadata to a UI-neutral overlay description."""
    return {
        "bbox": detection.get("bbox_xyxy"),
        "label": detection.get("class_name", "person"),
        "confidence": detection.get("confidence", 0.0),
    }

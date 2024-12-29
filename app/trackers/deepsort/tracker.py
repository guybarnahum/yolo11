from deep_sort_realtime.deepsort_tracker import DeepSort, Detection

# Initialize DeepSort tracker
deepsort_tracker = DeepSort(max_age=150) # 150 frames or 5 sec memory

def yolo_to_ltwh(xyxy):
    """Convert [x1, y1, x2, y2] bounding box to [left, top, width, height] format."""
    x1, y1, x2, y2 = xyxy
    left = x1
    top = y1
    width = x2 - x1
    height = y2 - y1
    return [left, top, width, height]


def track(detections, frame):
    
    # Create a DeepSort detection object
    detections_ds = []
    others = []

    for idx, detection in enumerate(detections):

        detection.track_id = None
        bbox_ltwh = yolo_to_ltwh(detection.bbox)
        
        # Create a deepsort detection object and store the YOLO index in the 'others' field
        detection_ds = [bbox_ltwh, detection.conf, detection.cls_id]
        detections_ds.append(detection_ds)
        others.append({'idx': idx})

    # Update the tracker with the detections
    tracks = deepsort_tracker.update_tracks(detections_ds, frame=frame, others = others)
    
    # Update track_id into original detection
    for track in tracks:
        if track.is_confirmed():
            others = track.get_det_supplementary()
            if others: 
                track_id = track.track_id  # Unique track ID
                idx = others['idx']

                detections[idx].track_id = track_id

    return detections

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSORTWrapper:
    def __init__(self, max_age=30, max_iou_distance=0.7, max_cosine_distance=0.3):
        self.tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            n_init=3,
            nn_budget=100
        )

    def update(self, bboxes, scores, class_ids, frame):
        if len(bboxes) == 0:
            return np.array([]), []

        detections = []
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            detection = ([float(x1), float(y1), float(w), float(h)], float(score), int(class_id))
            detections.append(detection)

        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        track_boxes = []
        track_ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_box = track.to_tlbr()
            track_id = track.track_id
            track_boxes.append(track_box)
            track_ids.append(track_id)

        return np.array(track_boxes), track_ids
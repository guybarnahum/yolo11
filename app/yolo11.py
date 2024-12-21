import fiftyone as fo
from fiftyone.core.labels import Detections, Detection
import fiftyone.utils.ultralytics as fou

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from deepsort_tracker import DeepSORTWrapper

# Global tracker instance
deepsort_tracker = None

def process_one_frame(frame, detect_model, tile_model, tracker, tile):
    global deepsort_tracker
    
    # Initialize DeepSORT
    if deepsort_tracker is None:
        deepsort_tracker = DeepSORTWrapper()

    # Define target classes
    class_codes = [0,1,2,3,4,5,6,7,8]

    if tile:
        # Get sliced predictions
        result = get_sliced_prediction(
            image=frame,
            detection_model=tile_model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
    
    # Run YOLO detection WITHOUT tracking
    results = detect_model.predict(
        source=frame,
        classes=class_codes,
        verbose=False,
        stream=True  # Return a generator of Results objects
    )
    
    result = next(results)  # Get the result from the generator
    
    # Extract boxes, scores, and class IDs
    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
    else:
        boxes = np.array([])
        scores = np.array([])
        class_ids = np.array([], dtype=int)

    # Update tracks using DeepSORT
    if len(boxes) > 0:
        track_boxes, track_ids = deepsort_tracker.update(boxes, scores, class_ids, frame)
    else:
        track_boxes, track_ids = np.array([]), []

    # Annotate frame with tracking results
    annotator = Annotator(frame, line_width=2)
    
    for box, track_id, cls_id, conf in zip(track_boxes, track_ids, class_ids, scores):
        # Get class name and create label
        class_name = result.names[cls_id] if cls_id < len(result.names) else "Unknown"
        label = f"{class_name} {conf:.2f}"
        
        # Generate color based on track_id
        color = colors(int(track_id), True)
        
        # Draw box and label
        annotator.box_label(box, label, color=color)

    # Create FiftyOne detections
    detections_list = []
    for box, track_id, cls_id, conf in zip(track_boxes, track_ids, class_ids, scores):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        detection = Detection(
            label=result.names[cls_id],
            bounding_box=[x1, y1, w, h],
            confidence=conf,
            index=track_id  
        )
        detections_list.append(detection)

    return frame, detections_list

def setup_model(model_path, tile=None, image_size=1088):
    # Wrap the YOLO model with SAHI's detection model if needed
    tile_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.5,
        device='cpu',
        image_size=image_size
    ) if tile else None
    
    detect_model = YOLO(model_path, verbose=False)
    return detect_model, tile_model
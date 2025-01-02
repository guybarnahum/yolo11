import fiftyone as fo
import cv2
import numpy as np
from fiftyone.core.labels import Detections, Detection
import fiftyone.utils.ultralytics as fou
from trackers.deepsort.tracker import track as deepsort_track

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def annotate_frame(frame, results, alt_tracks=None, sam2_results=None):
    """
    Annotate frame with YOLO detections and SAM2 segmentations
    """
    annotator = Annotator(frame, line_width=2)

    # First draw SAM2 segmentation masks if available
    if sam2_results:
        for mask, box, conf, cls_id in sam2_results:
            # Convert boolean mask to uint8
            mask_img = (mask * 255).astype(np.uint8)
            
            # Find and draw contours
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Use a unique color for this detection
            color = colors(cls_id, True)
            
            # Draw contours
            cv2.drawContours(frame, contours, -1, color, 2)

    # Then draw YOLO boxes and labels
    for idx, result in enumerate(results):
        try:
            track_ids = alt_tracks[idx] if alt_tracks else result.boxes.id.cpu().tolist()
        except Exception as e:
            track_ids = None

        if not track_ids:
            continue

        try:
            masks = result.masks.xy 
        except Exception as e:
            masks = None 
        
        for ix, track_id in enumerate(track_ids):
            conf = result.boxes.conf[ix].cpu().item()
            cls_id = int(result.boxes.cls[ix].cpu().item())
            class_label = result.names[cls_id] if cls_id < len(result.names) else "Unknown"
            label = f"{class_label} {conf:.2f}"
            color = colors(int(track_id), True)
    
            # Draw bounding box
            x1, y1, x2, y2 = result.boxes.xyxy[ix].cpu().tolist()
            annotator.box_label([x1, y1, x2, y2], label, color=color)

    return frame

def process_one_frame(frame, detect_model, tile_model, tracker, tile, sam2_handler=None):
    """
    Process one frame with YOLO detection and optional SAM2 segmentation
    """
    class_codes = [0,1,2,3,4,5,6,7,8]
 
    if tile:
        result = get_sliced_prediction(
            image=frame, 
            detection_model=tile_model,
            slice_height=tile, 
            slice_width=tile,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

    if tracker == 'deepsort':
        results = detect_model.predict(
            source=frame,
            classes=class_codes,
            verbose=False
        )
        alt_tracks = deepsort_track(results, frame)
    else:
        results = detect_model.track(
            frame, 
            classes=class_codes,
            persist=True,
            verbose=False,
            tracker=tracker
        ) 
        alt_tracks = None

    # Get SAM2 segmentation masks if handler is provided
    sam2_results = []
    if sam2_handler is not None:
        try:
            sam2_results = sam2_handler.process_yolo_detections(frame, results)
        except Exception as e:
            logging.error(f"Error getting SAM2 segmentations: {e}")

    # Annotate frame with both YOLO and SAM2 results
    frame = annotate_frame(frame, results, alt_tracks, sam2_results)
  
    # Convert YOLO results to FiftyOne Detections
    flatten_results = []
    for index, result in enumerate(results):
        flatten_results.extend(result)
        
    detections_obj = fou.to_detections(flatten_results)
    
    detections_list = []
    for d in detections_obj:
        detections_list.extend(d.detections)

    return frame, detections_list

def setup_model(model_path, tile=None, image_size=1088):
    """Setup YOLO model for detection"""
    tile_model = None
    if tile:
        tile_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path, 
            confidence_threshold=0.5,
            device='cpu',
            image_size=image_size
        )
    
    detect_model = YOLO(model_path, verbose=False)
    detect_model.cpu()  # Ensure model is on CPU

    return detect_model, tile_model
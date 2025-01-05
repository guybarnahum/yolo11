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

def annotate_frame(frame, results, alt_tracks=None, sam2_handler=None, sam2_results=None):
    """
    Annotate frame with YOLO detections and SAM2 segmentations
    """
    frame_with_masks = frame.copy()
    annotator = Annotator(frame_with_masks, line_width=2)

    # First draw SAM2 segmentation masks if available
    if sam2_results and sam2_handler:
        for mask, box, conf, cls_id, track_id in sam2_results:
            if mask is None:
                continue
                
            # Convert boolean mask to uint8 for visualization
            mask_img = (mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            # Get consistent color for this tracked object
            color = sam2_handler.get_color_for_id(track_id)
            
            # Create semi-transparent mask overlay
            mask_overlay = np.zeros_like(frame_with_masks)
            cv2.fillPoly(mask_overlay, contours, color)
            alpha = 0.5
            frame_with_masks = cv2.addWeighted(frame_with_masks, 1.0, mask_overlay, alpha, 0)
            
            # Draw contour edges
            cv2.drawContours(frame_with_masks, contours, -1, color, 2)
            
            # Draw box if available
            if box is not None and conf is not None:
                x1, y1, x2, y2 = box.astype(int)
                class_name = "person" if cls_id == 0 else "car"
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                annotator.box_label([x1, y1, x2, y2], label, color=color)

    # Then draw YOLO boxes and labels if needed
    if results and not sam2_results:  # Only draw YOLO if no SAM2 results
        for idx, result in enumerate(results):
            boxes = result.boxes
            if not boxes:
                continue

            try:
                track_ids = alt_tracks[idx] if alt_tracks else boxes.id.cpu().tolist()
            except:
                track_ids = [0] * len(boxes)

            for ix, track_id in enumerate(track_ids):
                if track_id == 0:  # Skip untracked objects
                    continue
                    
                box = boxes.xyxy[ix].cpu().numpy()
                conf = boxes.conf[ix].cpu().item()
                cls_id = int(boxes.cls[ix].cpu().item())
                
                class_name = "person" if cls_id == 0 else "car" if cls_id == 2 else "unknown"
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                color = colors(int(track_id), True)
        
                # Draw bounding box
                x1, y1, x2, y2 = box.astype(int)
                annotator.box_label([x1, y1, x2, y2], label, color=color)

    return frame_with_masks

def process_one_frame(frame, detect_model, tile_model, tracker, tile, sam2_handler=None):
    """Process one frame with YOLO detection and optional SAM2 segmentation"""
    class_codes = [0, 2]  # person and car
 
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
            source=frame, 
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
            # Apply SAM2 segmentation directly
            frame = sam2_handler.apply_segmentation(frame, sam2_results)
        except Exception as e:
            logging.error(f"Error getting SAM2 segmentations: {e}")
    else:
        # Fall back to YOLO boxes if SAM2 not available
        frame = annotate_frame(frame, results, alt_tracks)
  
    # Convert YOLO results to FiftyOne Detections for dataset
    detections_list = []
    for result in results:
        detections = fou.to_detections([result])
        if detections:
            detections_list.extend(detections[0].detections)

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
    detect_model.to('cpu')  # Ensure model is on CPU

    return detect_model, tile_model
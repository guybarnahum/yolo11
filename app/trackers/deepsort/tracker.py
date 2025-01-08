from deep_sort_realtime.deepsort_tracker import DeepSort, Detection
from .osnet_embedder import OSNetEmbedder
import logging
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepsort_tracker')

# Global tracker instance
_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        try:
            # Path to OSNet model
            model_path = "./models/osnet_x1_0_imagenet.pt"
            
            if os.path.exists(model_path):
                logger.info("Initializing DeepSORT with OSNet embedder")
                embedder = OSNetEmbedder(
                    model_path=model_path,
                    use_cuda=True
                )
            else:
                logger.warning("OSNet model not found, falling back to MobileNet")
                embedder = "mobilenet"
                
            _tracker = DeepSort(
                max_age=150,
                embedder=embedder,
                half=True,
                bgr=True,
            )
        except Exception as e:
            logger.error(f"Error initializing OSNet embedder: {e}. Falling back to MobileNet")
            _tracker = DeepSort(
                max_age=150,
                embedder="mobilenet",
                half=True,
                bgr=True,
            )
    return _tracker

def yolo_to_ltwh(xyxy):
    """Convert [x1, y1, x2, y2] bounding box to [left, top, width, height] format."""
    try:
        if isinstance(xyxy, (list, tuple)):
            x1, y1, x2, y2 = xyxy
        elif isinstance(xyxy, np.ndarray):
            x1, y1, x2, y2 = xyxy.tolist()
        else:
            # Assume it's a tensor
            x1, y1, x2, y2 = xyxy.cpu().tolist()
        return [x1, y1, x2-x1, y2-y1]
    except Exception as e:
        logger.error(f"Error converting bbox format: {e}")
        logger.error(f"Input bbox: {xyxy}, type: {type(xyxy)}")
        raise

def track(results, frame):
    try:
        # Get or create tracker instance
        deepsort_tracker = get_tracker()
        
        # Process detections
        detections = []
        others = []
        
        for idx, result in enumerate(results):
            try:
                boxes = result.boxes
                for jdx in range(len(boxes.xyxy)):
                    try:
                        bbox_xyxy = boxes.xyxy[jdx].cpu().tolist()
                        confidence = boxes.conf[jdx].cpu().item()
                        class_id = int(boxes.cls[jdx].cpu().item())
                        
                        bbox_ltwh = yolo_to_ltwh(bbox_xyxy)
                        detection = [bbox_ltwh, confidence, class_id]
                        detections.append(detection)
                        others.append({'idx': idx, 'jdx': jdx})
                        
                    except Exception as e:
                        logger.error(f"Error processing individual detection: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing result {idx}: {e}")
                continue

        # Update tracks
        tracks = deepsort_tracker.update_tracks(detections, frame=frame, others=others)
        
        # Initialize track IDs array
        track_ids = []
        for ix, result in enumerate(results):
            track_ids.append([0] * len(result.boxes.xyxy))
                
        # Update track IDs
        for track in tracks:
            if track.is_confirmed():
                others = track.get_det_supplementary()
                if others:
                    track_id = track.track_id
                    idx = others['idx']
                    jdx = others['jdx']
                    track_ids[idx][jdx] = track_id
                    
        return track_ids
        
    except Exception as e:
        logger.error(f"Error in DeepSort tracking: {e}")
        # Return empty tracking results on error
        return [[0] * len(result.boxes.xyxy) for result in results]
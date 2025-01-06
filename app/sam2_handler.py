from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
import logging
import os
import traceback
import colorsys
import cv2
from collections import deque

class TrackInfo:
    def __init__(self, mask, box, cls_id, max_history=10):
        self.mask_history = deque([mask], maxlen=max_history)
        self.box_history = deque([box], maxlen=max_history)
        self.cls_id = cls_id
        self.last_seen = 0
        self.velocity = np.zeros(4)  # box velocity
        
    def update(self, mask, box):
        self.mask_history.append(mask)
        if len(self.box_history) > 0:
            # Update velocity using box centers
            prev_box = self.box_history[-1]
            prev_center = ((prev_box[0] + prev_box[2])/2, (prev_box[1] + prev_box[3])/2)
            curr_center = ((box[0] + box[2])/2, (box[1] + box[3])/2)
            self.velocity = np.array([
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1],
                box[2] - box[0] - (prev_box[2] - prev_box[0]),  # width change
                box[3] - box[1] - (prev_box[3] - prev_box[1])   # height change
            ])
        self.box_history.append(box)
        
    def predict_box(self):
        if len(self.box_history) == 0:
            return None
        last_box = self.box_history[-1]
        # Predict next box using velocity
        center_x = (last_box[0] + last_box[2])/2 + self.velocity[0]
        center_y = (last_box[1] + last_box[3])/2 + self.velocity[1]
        width = last_box[2] - last_box[0] + self.velocity[2]
        height = last_box[3] - last_box[1] + self.velocity[3]
        return np.array([
            center_x - width/2,
            center_y - height/2,
            center_x + width/2,
            center_y + height/2
        ])

class SAM2Handler:
    def __init__(self, model_path="./models/sam2_hiera_t.pt"):
        """Initialize SAM2 handler for video processing"""
        try:
            # Determine config path based on model name
            model_name = os.path.basename(model_path)
            if "tiny" in model_name or "_t" in model_name:
                model_cfg = "configs/sam2/sam2_hiera_t.yaml"
            elif "small" in model_name or "_s" in model_name:
                model_cfg = "configs/sam2/sam2_hiera_s.yaml"
            elif "base" in model_name or "_b+" in model_name:
                model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
            elif "large" in model_name or "_l" in model_name:
                model_cfg = "configs/sam2/sam2_hiera_l.yaml"
            else:
                model_cfg = "configs/sam2/sam2_hiera_t.yaml"
                
            logging.info(f"Using SAM2 config: {model_cfg}")
            
            # Initialize SAM2 model on CPU
            self.model = build_sam2(model_cfg, model_path, device="cpu")
            self.predictor = SAM2ImagePredictor(self.model)
            self.object_colors = {}
            self.next_id = 1
            self.tracks = {}  # track_id -> TrackInfo
            self.max_frames_missing = 30  # Allow tracks to coast this many frames
            logging.info("SAM2 initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize SAM2: {e}")
            logging.error(traceback.format_exc())
            raise e

    def get_unique_color(self, track_id):
        """Generate a unique color for each track ID"""
        if track_id not in self.object_colors:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = tuple(int(255 * c) for c in rgb[::-1])
            self.object_colors[track_id] = color
        return self.object_colors[track_id]

    def _calculate_matching_score(self, mask1, box1, mask2, box2):
        """Calculate matching score between two detections using multiple metrics"""
        # IoU of masks
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        mask_iou = intersection / union if union > 0 else 0
        
        # IoU of boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            box_iou = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box_iou = intersection_area / (box1_area + box2_area - intersection_area)
        
        # Center distance score
        c1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
        c2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
        center_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        max_dist = np.sqrt(mask1.shape[0]**2 + mask1.shape[1]**2)
        center_score = 1 - min(center_dist / max_dist, 1.0)
        
        # Size similarity score
        size1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        size2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        size_ratio = min(size1, size2) / max(size1, size2)
        
        # Weighted combination
        return 0.4 * mask_iou + 0.3 * box_iou + 0.2 * center_score + 0.1 * size_ratio

    def _track_objects(self, current_frame, current_masks):
        """Match current masks with existing tracks using multiple cues"""
        if not self.tracks:
            # First frame - create new tracks
            tracked_masks = []
            for mask, box, conf, cls_id in current_masks:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = TrackInfo(mask, box, cls_id)
                tracked_masks.append((mask, box, conf, cls_id, track_id))
            return tracked_masks

        # Predict track locations
        track_predictions = {
            track_id: track.predict_box() 
            for track_id, track in self.tracks.items()
        }

        # Calculate matching scores
        matching_scores = {}
        for curr_idx, (curr_mask, curr_box, _, curr_cls) in enumerate(current_masks):
            for track_id, track in self.tracks.items():
                if track.cls_id != curr_cls:
                    continue
                    
                pred_box = track_predictions[track_id]
                if pred_box is None:
                    continue
                    
                score = self._calculate_matching_score(
                    curr_mask, curr_box,
                    track.mask_history[-1], pred_box
                )
                if score > 0.3:  # Lower threshold for matching
                    matching_scores[(curr_idx, track_id)] = score

        # Greedy matching
        matched_curr = set()
        matched_tracks = set()
        tracked_masks = []

        # Sort matches by score
        sorted_matches = sorted(matching_scores.items(), key=lambda x: x[1], reverse=True)
        
        for (curr_idx, track_id), score in sorted_matches:
            if curr_idx in matched_curr or track_id in matched_tracks:
                continue
                
            curr_mask, curr_box, curr_conf, curr_cls = current_masks[curr_idx]
            self.tracks[track_id].update(curr_mask, curr_box)
            self.tracks[track_id].last_seen = current_frame
            
            tracked_masks.append((curr_mask, curr_box, curr_conf, curr_cls, track_id))
            matched_curr.add(curr_idx)
            matched_tracks.add(track_id)

        # Create new tracks for unmatched detections
        for curr_idx, (curr_mask, curr_box, curr_conf, curr_cls) in enumerate(current_masks):
            if curr_idx in matched_curr:
                continue
                
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = TrackInfo(curr_mask, curr_box, curr_cls)
            self.tracks[track_id].last_seen = current_frame
            tracked_masks.append((curr_mask, curr_box, curr_conf, curr_cls, track_id))

        # Remove old tracks
        self.tracks = {
            track_id: track 
            for track_id, track in self.tracks.items()
            if current_frame - track.last_seen <= self.max_frames_missing
        }

        return tracked_masks

    def process_yolo_detections(self, frame, yolo_results, frame_idx=0):
        """Process YOLO detections with SAM2 segmentation and tracking"""
        try:
            current_masks = []
            person_car_classes = [0, 2]  # YOLO class IDs for person and car
            
            with torch.inference_mode():
                self.predictor.set_image(frame)
                
                for result in yolo_results:
                    for i, box in enumerate(result.boxes.xyxy):
                        cls_id = int(result.boxes.cls[i].cpu().item())
                        if cls_id not in person_car_classes:
                            continue
                            
                        box_np = box.cpu().numpy()
                        conf = result.boxes.conf[i].cpu().item()
                        
                        try:
                            masks, _, _ = self.predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box_np[None, :],
                                multimask_output=False
                            )
                            
                            if masks is not None and len(masks) > 0:
                                current_masks.append((
                                    masks[0],
                                    box_np,
                                    conf,
                                    cls_id
                                ))
                                
                        except Exception as e:
                            logging.error(f"Error processing individual detection: {e}")
                            continue
                
                # Track objects using multiple cues
                tracked_results = self._track_objects(frame_idx, current_masks)
                return tracked_results
                
        except Exception as e:
            logging.error(f"Error in process_yolo_detections: {e}")
            logging.error(traceback.format_exc())
            return []

    def apply_segmentation(self, frame, results):
        """Apply segmentation masks to frame with unique colors"""
        frame_with_masks = frame.copy()
        
        if not results:
            return frame_with_masks
            
        for mask, box, conf, cls_id, track_id in results:
            if mask is None:
                continue
                
            color = self.get_unique_color(track_id)
            mask_img = mask.astype(np.uint8) * 255
            
            mask_overlay = np.zeros_like(frame_with_masks)
            mask_indices = np.where(mask)
            mask_overlay[mask_indices] = color
            
            alpha = 0.5
            cv2.addWeighted(mask_overlay, alpha, frame_with_masks, 1, 0, frame_with_masks)
            
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_with_masks, contours, -1, color, 2)
            
            x1, y1, x2, y2 = box.astype(int)
            class_name = "person" if cls_id == 0 else "car"
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_with_masks, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            cv2.putText(frame_with_masks, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return frame_with_masks

    def cleanup(self):
        """Clean up resources"""
        self.predictor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
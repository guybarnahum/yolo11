from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
import logging
import os
import traceback
import colorsys
import cv2

class SAM2Handler:
    def __init__(self, model_path="./models/sam2_hiera_t.pt"):
        """Initialize SAM2 handler for image processing"""
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
            
            # Initialize  SAM2 model
            self.model = build_sam2(model_cfg, model_path, device="cpu")
            self.predictor = SAM2ImagePredictor(self.model)
            self.next_id = 1
            self.object_colors = {}
            logging.info("SAM2 initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize SAM2: {e}")
            logging.error(traceback.format_exc())
            raise e

    def get_unique_color(self, track_id):
        """Generate a unique color for each track ID"""
        if track_id not in self.object_colors:
            # Spread colors evenly
            hue = (track_id * 0.618033988749895) % 1.0
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            # Convert to BGR for OpenCV
            color = tuple(int(255 * c) for c in rgb[::-1])
            self.object_colors[track_id] = color
        return self.object_colors[track_id]

    def process_yolo_detections(self, frame, yolo_results):
        """Process YOLO detections with SAM2 for improved segmentation"""
        try:
            all_results = []
            person_car_classes = [0, 2]  # YOLO class IDs for person and car
            
            with torch.inference_mode():
                self.predictor.set_image(frame)
                
                for result in yolo_results:
                    for i, box in enumerate(result.boxes.xyxy):
                        cls_id = int(result.boxes.cls[i].cpu().item())
                        
                        # Only process people and cars
                        if cls_id not in person_car_classes:
                            continue
                            
                        box_np = box.cpu().numpy()
                        conf = result.boxes.conf[i].cpu().item()
                        
                        try:
                            # Get segmentation mask from SAM2
                            masks, _, _ = self.predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box_np[None, :],
                                multimask_output=False  # Get single best mask
                            )
                            
                            if masks is not None and len(masks) > 0:
                                mask = masks[0]  # Take first mask since multimask_output=False
                                
                                # Use YOLO's tracking ID if available
                                try:
                                    track_id = int(result.boxes.id[i].cpu().item())
                                except:
                                    # Fall back to our counter if no YOLO tracking ID
                                    track_id = self.next_id
                                    self.next_id += 1
                                
                                all_results.append((
                                    mask,
                                    box_np,
                                    conf,
                                    cls_id,
                                    track_id
                                ))
                                
                        except Exception as e:
                            logging.error(f"Error processing individual detection: {e}")
                            continue
                
                return all_results
                
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
                
            # Get unique color for this track
            color = self.get_unique_color(track_id)
            
            # Convert boolean mask to uint8
            mask_img = mask.astype(np.uint8) * 255
            
            # Create colored mask overlay
            mask_overlay = np.zeros_like(frame_with_masks)
            # Convert boolean mask to proper index array
            mask_indices = np.where(mask)
            mask_overlay[mask_indices] = color
            
            # Apply mask with alpha blending
            alpha = 0.5
            cv2.addWeighted(mask_overlay, alpha, frame_with_masks, 1, 0, frame_with_masks)
            
            # Draw contours
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_with_masks, contours, -1, color, 2)
            
            # Draw bounding box and label
            x1, y1, x2, y2 = box.astype(int)
            class_name = "person" if cls_id == 0 else "car"
            label = f"{class_name} {conf:.2f}"
            
            # Draw semi-transparent label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_with_masks, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame_with_masks, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return frame_with_masks

    def cleanup(self):
        """Clean up resources"""
        self.predictor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
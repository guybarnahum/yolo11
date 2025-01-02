from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
import logging

class SAM2Handler:
    def __init__(self, model_path="./models/sam2.1_hiera_tiny.pt", model_cfg="/app/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"):
        """Initialize SAM2 handler for CPU usage"""
        try:
            self.model = build_sam2(model_cfg, model_path)
            self.model = self.model.cpu()  # Ensure model is on CPU
            self.predictor = SAM2ImagePredictor(self.model)
            logging.info("SAM2 initialized successfully on CPU")
        except Exception as e:
            logging.error(f"Failed to initialize SAM2: {e}")
            raise e
        
    def process_yolo_detections(self, frame, yolo_results):
        """
        Process YOLO detections with SAM2 for improved segmentation
        Args:
            frame: numpy array of the current frame
            yolo_results: YOLO detection results
        Returns:
            List of tuples (mask, box, confidence, class_id)
        """
        try:
            with torch.inference_mode():
                self.predictor.set_image(frame)
                all_results = []
                
                for result in yolo_results:
                    for i, box in enumerate(result.boxes.xyxy):
                        # Get box info
                        box_np = box.cpu().numpy()
                        conf = result.boxes.conf[i].cpu().item()
                        cls_id = int(result.boxes.cls[i].cpu().item())
                        
                        # Get segmentation mask from SAM2
                        masks, _, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box_np[None, :],
                            multimask_output=False
                        )
                        
                        if masks is not None and len(masks) > 0:
                            all_results.append((masks[0], box_np, conf, cls_id))
                
                return all_results
        except Exception as e:
            logging.error(f"Error in process_yolo_detections: {e}")
            return []

    def cleanup(self):
        """Clean up resources"""
        self.predictor = None
        self.model = None
        torch.cuda.empty_cache()  # Just in case, though we're on CPU
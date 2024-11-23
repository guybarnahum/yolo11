from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def annotate_frame(frame, results):

     # initialize annotator for plotting masks
    annotator = Annotator(frame, line_width=2)  

    if results[0].boxes.id is not None :

        # Loop through each detected object (in the results)
        for result in results:
            
            # Extract list of track_ids
            track_ids = result.boxes.id.cpu().tolist()
            try:
                masks = result.masks.xy 
            except Exception as e:
                masks = None 

            if not track_ids:  # Skip if no track ID is present
                continue
            
            for ix, track_id in enumerate(track_ids):
                
                # Extract bounding box coordinates, confidence, and class ID
                conf = result.boxes.conf[ix].cpu().item()  # Confidence score
                cls_id = int(result.boxes.cls[ix].cpu().item())  # Class ID

                # Get the class name based on the class ID
                class_label = result.names[cls_id] if cls_id < len(result.names) else "Unknown"
                label = f"{class_label} {conf:.2f}"

                # Generate a color based on the track_id
                color = colors(int(track_id), True)
        
                # Draw the bounding box and the label with the generated color
                mask = masks[ix] if masks and ix < len(masks) else None
                
                if mask is not None:
                #if mask : # has mask? draw mask
                    annotator.seg_bbox(mask=mask, mask_color=color, label=label, txt_color=annotator.get_txt_color(color))
                else: # no mask - do box
                    x1, y1, x2, y2 = result.boxes.xyxy[ix].cpu().tolist()  # Assuming result has `.boxes.xyxy`
                    annotator.box_label([x1, y1, x2, y2], label, color=color)

    return frame


def process_one_frame( frame, detect_model, tile_model, tracker, tile ):
    
    class_codes = [0,1,2,3,4,5,6,7,8]
 
    if not tracker: 
        tracker="botsort.yaml"

    if tile:
         # Get sliced predictions
        result = get_sliced_prediction( image=frame, detection_model=tile_model,
                                        slice_height=tile, slice_width=tile,
                                        overlap_height_ratio=0.2,overlap_width_ratio=0.2
                                    )

    # object detection and tracking
    results = detect_model.track( frame, 
                                  classes=class_codes,
                                  persist=True,
                                  verbose=False,
                                  tracker=tracker
                                ) 

    #print(json.dumps(results.__dict__,indent=2))
    frame = annotate_frame(frame, results)
    
    return frame


def setup_model(model_path, tile=None, image_size=1088):

    # Wrap the YOLO model with SAHI's detection model
    tile_model = AutoDetectionModel.from_pretrained( model_type= 'yolov8', model_path=model_path, 
                                                     confidence_threshold=0.5,
                                                     device='cpu',
                                                     image_size=image_size
                                                    ) if tile else None
    
    detect_model = YOLO(model_path, verbose=False) 

    return detect_model, tile_model

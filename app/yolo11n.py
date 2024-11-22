
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging
import json

def process_one_frame( frame, detect_model, track_model ):
    
    class_codes = [0,1,2,3,4,5,6,7,8]

    # initialize annotator for plotting masks
    annotator = Annotator(frame, line_width=2)  
    
    # object tracking
    results   = track_model.track(  frame, 
                                    classes=class_codes,
                                    persist=True,
                                    verbose=False,
                                    #tracker="bytetrack.yaml"
                                    tracker="botsort.yaml"
                                    ) 

    #print(json.dumps(results.__dict__,indent=2))

    if results[0].boxes.id is not None :

        # Loop through each detected object (in the results)
        for result in results:
            
            # Extract list of track_ids
            track_ids = result.boxes.id.cpu().tolist() 
            if not track_ids:  # Skip if no track ID is present
                continue
            
            for ix, track_id in enumerate(track_ids):
                
                # Extract bounding box coordinates, confidence, and class ID
                x1, y1, x2, y2 = result.boxes.xyxy[ix].cpu().tolist()  # Assuming result has `.boxes.xyxy`
                conf = result.boxes.conf[ix].cpu().item()  # Confidence score
                cls_id = int(result.boxes.cls[ix].cpu().item())  # Class ID

                # Get the class name based on the class ID
                class_label = result.names[cls_id] if cls_id < len(result.names) else "Unknown"
                label = f"{class_label} {conf:.2f}"

                # print(f"{ix}>>>> {label}")

                # Generate a color based on the track_id
                color = colors(int(track_id), True)

                # Draw the bounding box and the label with the generated color
                annotator.box_label([x1, y1, x2, y2], label, color=color)

    return frame

def setup_model( model_path, image_size ):

    model = YOLO(model_path,verbose=False) 
    return model

def get_models(image_size=1088):
    detect_model = YOLO("./models/yolo11n.pt",
                        verbose=False
                        ) 
    track_model  = detect_model

    return detect_model, track_model

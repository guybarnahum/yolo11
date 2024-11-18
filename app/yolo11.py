
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging

def process_one_frame( frame, detect_model, track_model ):
    
    # initialize annotator for plotting masks
    annotator = Annotator(frame, line_width=2)  
    results   = track_model.track(frame, persist=True,verbose=False) # object tracking
        
    # check if tracks and masks are not None, then plot the masks on frame
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
            
        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color,
                            label=str(track_id),
                            txt_color=txt_color)

    return frame

def setup_model( model_path, image_size ):

    model = YOLO(model_path,verbose=False) 
    return model

def get_models(image_size=1088):
    detect_model = YOLO("./models/yolo11n-seg.pt",verbose=False) 
    track_model  = detect_model

    return detect_model, track_model

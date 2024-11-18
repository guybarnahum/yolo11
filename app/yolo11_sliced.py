# import cv2
import logging

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def process_one_frame( frame, detect_model, track_model ):
    annotator = Annotator(frame, line_width=2)  
    # Get sliced predictions
    result = get_sliced_prediction(
        image=frame,
        detection_model=detect_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Run tracking on the original image
    tracked_objects = track_model.track(frame, verbose=False, classes=0,conf=0.5,persist=True)[0]
    # print('tracked_objects:', tracked_objects[0].boxes.xyxy)
    # Draw bounding boxes on the frame
    # check if tracks and masks are not None, then plot the masks on frame
    if tracked_objects.boxes.id is not None:
        masks = tracked_objects.masks.xy
        track_ids = tracked_objects.boxes.id.int().cpu().tolist()
            
        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color,
                            label=str(track_id),
                            txt_color=txt_color)

    # for box in tracked_objects.boxes.xyxy:
    #   box = box.cpu().numpy().astype(int)
    #   cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    return frame

def setup_model( model_path, image_size=1088 ):

    # Wrap the YOLO model with SAHI's detection model
    detect_model = AutoDetectionModel.from_pretrained(
        model_type= 'yolov8',
        model_path=model_path,
        confidence_threshold=0.3,
        device='cpu',
        image_size=image_size
    )

    return detect_model


def get_models(image_size=1088):
    detect_model = setup_model("./models/yolo11n.pt", image_size=image_size) 
    track_model  = YOLO("./models/yolo11n-seg.pt")

    return detect_model, track_model

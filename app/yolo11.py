from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from types import SimpleNamespace

from trackers.deepsort.tracker import track as deepsort_track

cls_id_car_type = ["bicycle","motorcycle","bus","train","truck","boat","van"]
cls_id_car_list = []
cls_id_car = None

def map_cls_id_build( cls_dict ):
    '''
    Yolo sometimes confuses cars with bicycle, motorcycle, etc 
    '''
    global cls_id_car_type
    global cls_id_car
    global cls_id_car_list
    
    logging.info( f"model classes: {cls_dict}")

    for cls_id, name in cls_dict.items():

        if name == "car":  cls_id_car = cls_id
        if name in cls_id_car_type: cls_id_car_list.append(cls_id)

    if not cls_id_car:
        logging.error(f"Could not locate `car` in model classes")
    
    if not cls_id_car:
        logging.warning(f"Could not locate class_ids for {cls_id_car_type}")
    
    logging.info(f"mapping {cls_id_car_list} to car cls_id : {cls_id_car}")

def map_cls_id( cls_id ):
    
    global cls_id_car
    global cls_id_car_list

    # yolo sometimes confuses cars with bicycle, motorcycle, etc 
    if cls_id in cls_id_car_list:
        cls_id = cls_id_car

    return cls_id

def flatten_results(results):

    detections = []

    for idx, cls_results in enumerate(results):
        
        try:
            track_ids = cls_results.boxes.id.cpu().tolist()
        except Exception as e:
            # logging.warning(f"Getting track_ids failed : {str(e)}")
            track_ids =  None

        try:
            masks =  cls_results.masks.xy
        except Exception as e:
            masks = None

        for jdx, box in enumerate(cls_results.boxes):
            
            bbox   = cls_results.boxes.xyxy[jdx].cpu().tolist()
            conf   = cls_results.boxes.conf[jdx].cpu().item()  # Confidence score
            cls_id = int(cls_results.boxes.cls[jdx].cpu().item())  # Class ID
            cls_id = map_cls_id(cls_id)

            try:
                name = cls_results.names[ cls_id ] if cls_id < len(cls_results.names) else "Unknown"
            except Exception as e:
                name = "Unknown"

            mask        = masks[jdx]     if masks else None
            track_id    = int(track_ids[jdx]) if track_ids else None

            detection = SimpleNamespace()
            detection.bbox = bbox
            detection.conf = conf
            detection.cls_id = cls_id
            detection.name = name
            detection.mask = mask
            detection.track_id = track_id

            detections.append( detection )

    return detections


def annotate_frame(frame, detections, frame_number=None):
    # initialize annotator for plotting masks
    annotator = Annotator(frame, line_width=2)

    for idx, detection in enumerate(detections):

        # Get the class name based on the class ID
        try:
            class_label = detection.name or "Unknown"
        except Exception as e:
            print(str(e))
            class_label = "Unknown"

        label = f"{class_label} {detection.track_id} {detection.conf:.2f}"

        # Generate a color based on the track_id
        track_id = detection.track_id if detection.track_id else 0
        color = colors(track_id, True)
    
        if detection.mask is not None: # has mask? draw mask
            annotator.seg_bbox(mask=detection.mask, mask_color=color, label=label, txt_color=annotator.get_txt_color(color))
        else: # no mask - do box
            x1, y1, x2, y2 = detection.bbox  
            annotator.box_label([x1, y1, x2, y2], label, color=color)

    if frame_number:
        # Define the position and text for the frame number
        position = (10, 50)  # (x, y) position on the frame
        text = f"Frame: {frame_number}"

        # Draw the text on the frame
        annotator.text(position, text,txt_color=(0,0,0),box_style=True)

    return frame


def process_one_frame( frame, detect_model, tile_model, tracker, tile, frame_number = None ):

    # 0: "person"
    # 1: "bicycle"
    # 2: "car"
    # 3: "motorcycle"
    # 4: "airplane"
    # 5: "bus"
    # 6: "train"
    # 7: "truck"
    # 8: "boat"

    class_codes = [0,1,2,3,4,5,6,7,8]
 
    if tile:
         # Get sliced predictions
        results = get_sliced_prediction( image=frame, detection_model=tile_model,
                                        slice_height=tile, slice_width=tile,
                                        overlap_height_ratio=0.2,overlap_width_ratio=0.2
                                    )

    # object detection and tracking
    if tracker == 'deepsort' :
        results = detect_model.predict( source=frame,
                                        classes=class_codes,
                                        verbose=False
                                    )
        detections = flatten_results(results)
        detections = deepsort_track(detections, frame)

    else:
        results = detect_model.track( frame, 
                                      classes=class_codes,
                                      persist=True,
                                      verbose=False,
                                      tracker=tracker
                                    ) 
        detections = flatten_results(results)

    frame = annotate_frame(frame, detections, frame_number=frame_number)
  
    return frame, detections


def setup_model(model_path, tile=None, image_size=1088):

    # Wrap the YOLO model with SAHI's detection model
    tile_model = AutoDetectionModel.from_pretrained( model_type= 'yolov8', model_path=model_path, 
                                                     confidence_threshold=0.5,
                                                     device='cpu',
                                                     image_size=image_size
                                                    ) if tile else None
    
    detect_model = YOLO(model_path, verbose=False) 
    
    map_cls_id_build( detect_model.names )
    return detect_model, tile_model

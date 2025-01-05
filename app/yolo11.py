from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from types import SimpleNamespace

from trackers.deepsort.tracker import track as deepsort_track

def bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    Returns:
        IoU value (float)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) to remove overlapping detections,
    optimized by sorting based on x-coordinates.

    Args:
        detections: List of detection objects, each with .bbox and .conf attributes.
        iou_threshold: IoU threshold to filter overlapping boxes.

    Returns:
        List of filtered detection objects.
    """
    if len(detections) == 0:
        return []

    # Step 1: Sort detections by the leftmost x-coordinate (bbox[0])
    detections = sorted(detections, key=lambda det: det.bbox[0])

    filtered_detections = []
    while detections:
        # Step 2: Take the detection with the highest confidence
        best_idx = max(range(len(detections)), key=lambda i: detections[i].conf)
        best_detection = detections.pop(best_idx)
        filtered_detections.append(best_detection)

        # Step 3: Compare only with neighboring bboxes based on x-coordinates
        neighbors = [
            det for det in detections 
            if det.bbox[0] <= best_detection.bbox[2]  # Neighbor starts before 'best' ends
        ]

        # Remove overlapping neighbors
        detections = [
            det for det in detections 
            if det not in neighbors or bbox_iou(best_detection.bbox, det.bbox) < iou_threshold
        ]
    
    return filtered_detections



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
        if name in cls_id_car_type and cls_id not in cls_id_car_list: cls_id_car_list.append(cls_id)

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

def flatten_results(results, min_conf = None):

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
            
            conf   = cls_results.boxes.conf[jdx].cpu().item()  # Confidence score

            # skip detections with low conf
            if min_conf and min_conf > conf:
                continue

            bbox   = cls_results.boxes.xyxy[jdx].cpu().tolist()
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

    detections = non_max_suppression(detections)
    return detections

def print_detections(detections, frame_number = None):
    
    if frame_number:
        print(f">>>>>>>>>>>>>>>>>>>>> Frame {frame_number} <<<<<<<<<<<<<<<<<<<<<")

    for d in detections:
        conf = round(d.conf,2)
        bbox = [round(num, 2) for num in d.bbox]

        print(f"{d.track_id},{d.name},{bbox},{conf}")


frames_to_debug = None # [1,2,3,4,5] 

def annotate_frame(frame, detections, frame_number=None):
    global frames_to_debug

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

    if frames_to_debug and frame_number in frames_to_debug:
        print_detections(detections,frame_number)

    return frame


def process_one_frame( frame, detect_model, tile_model, tracker, tile, conf, frame_number = None ):

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
    if not conf: conf = 0.3

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
                                        conf=conf,
                                        verbose=False
                                    )
        detections = flatten_results(results, conf)
        detections = deepsort_track(detections, frame)

    else:
        results = detect_model.track( frame, 
                                      classes=class_codes,
                                      conf=conf,
                                      persist=True,
                                      verbose=False,
                                      tracker=tracker
                                    ) 
        detections = flatten_results(results, conf)

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

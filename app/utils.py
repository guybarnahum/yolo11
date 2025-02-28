import logging
import json
import cv2
import os

import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from sahi import AutoDetectionModel

from types import SimpleNamespace, MethodType
from torch import cuda

def print_loggers(show_noset=False):
    logger_dict = logging.root.manager.loggerDict
    for logger_name, logger_obj in logger_dict.items():
        if isinstance(logger_obj, logging.Logger):  # Ensure it's a Logger instance
            level = logging.getLevelName(logger_obj.level)
            if level != 'NOTSET' or show_noset:
                print(f"Logger: {logger_name}, Level: {level}")

    root_level = logging.getLogger().level
    print(f"Logger Root, Level: {logging.getLevelName(root_level)}")


cuda_device_type = None

def cuda_device(force=False):

    global cuda_device_type

    if cuda_device_type and not force: 
        return cuda_device_type

    # force is set or cuda_device_type is unknown as None
    #  
    device = 'cpu' # default in case of errors
    try:
        if cuda.is_available():
            num_gpus = cuda.device_count()
            logging.info(f"cuda_device: GPU is available: {num_gpus} GPU(s) detected.")
            for i in range(num_gpus):
                print(f"  - GPU {i}: {cuda.get_device_name(i)}")
            device = 'cuda:0'  # Default to first GPU
        else:
            logging.info("cuda_device: No GPU available. Using CPU.")
            device = 'cpu'

    except Exception as e:
        logging.error(f"check_device Error: {e}")
        device = 'cpu'

    cuda_device_type = device
    return device

# Detect cuda device type
cuda_device_type = cuda_device()

class frameAnnotator(Annotator):
    def centered_text(self, bbox, text, txt_color=(0,0,0), box_style=False):
        """
        Draw text centered in the bounding box.

        Args:
            bbox: Bounding box coordinates as [x1, y1, x2, y2].
            text: The text to center in the bounding box.
            font: Font type for the text (default cv2.FONT_HERSHEY_SIMPLEX).
            font_scale: Scale of the font (default 0.5).
            thickness: Thickness of the text (default 1).
            txt_color: Color of the text in BGR (default white).
            box_style: (default False)
        """
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Measure text size
        font        = cv2.FONT_HERSHEY_SIMPLEX # Annotator hardcodes the font
        font_scale  = self.sf
        thickness   = self.tf

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate offsets for centering
        offset_x = text_width / 2
        offset_y = text_height / 2

        # Calculate text position
        text_x = int(center_x - offset_x)
        text_y = int(center_y + offset_y)  # Adjust for OpenCV baseline behavior

        # Draw the text
        self.text((text_x, text_y), text, txt_color=txt_color, box_style=box_style)


def coco_pose_keypoint_name(index):
    """
    Returns the name of the keypoint based on the COCO keypoint format.

    Args:
        index (int): Index of the keypoint (0-based).

    Returns:
        str: Name of the keypoint if valid, otherwise 'Invalid keypoint index'.
    """
    coco_keypoints = [
        "Nose", 
        "Left Eye", 
        "Right Eye", 
        "Left Ear", 
        "Right Ear", 
        "Left Shoulder", 
        "Right Shoulder", 
        "Left Elbow", 
        "Right Elbow", 
        "Left Wrist", 
        "Right Wrist", 
        "Left Hip", 
        "Right Hip", 
        "Left Knee", 
        "Right Knee", 
        "Left Ankle", 
        "Right Ankle"
    ]
    
    if 0 <= index < len(coco_keypoints):
        return coco_keypoints[index]
    else:
        return "Unknown"

COCO_SKELETON = [
    (0, 1),  # Nose -> Left Eye
    (0, 2),  # Nose -> Right Eye
    (1, 3),  # Left Eye -> Left Ear
    (2, 4),  # Right Eye -> Right Ear
    (0, 5),  # Nose -> Left Shoulder
    (0, 6),  # Nose -> Right Shoulder
    (5, 6),  # Left Shoulder -> Right Shoulder
    (5, 7),  # Left Shoulder -> Left Elbow
    (7, 9),  # Left Elbow -> Left Wrist
    (6, 8),  # Right Shoulder -> Right Elbow
    (8, 10), # Right Elbow -> Right Wrist
    (5, 11), # Left Shoulder -> Left Hip
    (6, 12), # Right Shoulder -> Right Hip
    (11, 12),# Left Hip -> Right Hip
    (11, 13),# Left Hip -> Left Knee
    (13, 15),# Left Knee -> Left Ankle
    (12, 14),# Right Hip -> Right Knee
    (14, 16) # Right Knee -> Right Ankle
]


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


def map_cls_id_build( cls_dict ):
    '''
    Yolo sometimes confuses cars with trucks, buses, etc 
    '''
    cls_id_car_types = [ "bus", "train", "truck", "boat", "van" ]
    cls_id_car       = None
    cls_id_car_list  = []

    cls_dict_trunc_str = str(cls_dict)[:125] + '...'
    logging.info( f"model classes: { cls_dict_trunc_str }")

    for cls_id, name in cls_dict.items():

        if name == "car":  cls_id_car = cls_id
        if name in cls_id_car_types and cls_id not in cls_id_car_list: cls_id_car_list.append(cls_id)

    if not cls_id_car:
        logging.warning(f"Could not locate `car` in model classes")
    
    if not cls_id_car_list:
        logging.warning(f"Could not locate class_ids for {cls_id_car_types}")
    
    logging.info(f"mapping {cls_id_car_list} to car cls_id : {cls_id_car}")
    return cls_id_car, cls_id_car_list


def map_cls_id( self, cls_id ):
    
    # yolo sometimes confuses cars with bicycle, motorcycle, etc 
    if cls_id in self.cls_id_car_list:
       cls_id = self.cls_id_car

    return cls_id


def setup_model(model_path, tile=None, image_size=108, build_cls_map=False):

    # Wrap the YOLO model with SAHI's detection model
    device = cuda_device()
    tile_model = AutoDetectionModel.from_pretrained( model_type= 'yolov8', model_path=model_path, 
                                                     confidence_threshold=0.5,
                                                     device=device,
                                                     image_size=image_size
                                                    ) if tile else None
    
    detect_model = YOLO(model_path, verbose=False)

    # Build car class-id list from detect_model.names
    if build_cls_map:
        detect_model.cls_id_car, detect_model.cls_id_car_list = map_cls_id_build( detect_model.names )
        detect_model.map_cls_id = MethodType(map_cls_id, detect_model)
    else:
        detect_model.map_cls_id = None

    return detect_model, tile_model


def flatten_results(results, min_conf = None, frame_number = None, should_inspect = None, map_cls_id = None, offset_x = None, offset_y = None):

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
            
            conf = round(cls_results.boxes.conf[jdx].cpu().item(),3)  # Confidence score

            # skip detections with low conf
            if min_conf and min_conf > conf:
                continue

            cls_id = int(cls_results.boxes.cls[jdx].cpu().item())  # Class ID
            cls_id = map_cls_id(cls_id) if map_cls_id else cls_id

            try:
                name = cls_results.names[ cls_id ] if cls_id < len(cls_results.names) else "Unknown"
            except Exception as e:
                name = "Unknown"

            x1, y1, x2, y2 = cls_results.boxes.xyxy[jdx].cpu().tolist()
            
            if offset_x:
                x1 = x1 + offset_x
                x2 = x2 + offset_x
            
            if offset_y:
                y1 = y1 + offset_y
                y2 = y2 + offset_y

            bbox   = [round(coord,2) for coord in [x1, y1, x2, y2]]
            
            area   = (x2 - x1) * (y2 - y1)
            if area < 0 : area = -area

            mask        = masks[jdx]     if masks else None
            track_id    = int(track_ids[jdx]) if track_ids else None

            detection = SimpleNamespace()
            detection.bbox = bbox
            detection.conf = conf
            detection.cls_id = cls_id
            detection.name = name
            detection.area = area
            detection.frame_number = frame_number
            detection.inspect = should_inspect(detection) if should_inspect else False
            detection.attributes = []
            detection.mask = mask
            detection.detail = None
            detection.track_id = track_id

            detections.append( detection )

    detections = non_max_suppression(detections)
    return detections


def print_detection( d , index = None):

    conf = float(d.conf)
    bbox = [round(num, 2) for num in d.bbox]
    detail = d.detail
    fn     = d.frame_number
    ix     = str(index) if index is not None else '-'

    if d.track_id:
        print(f"{ix}> track_id:{d.track_id},{d.name},[{bbox}],conf:{conf:.2f}, detail:{detail}, fn#:{fn}")
    else:
        print(f"{ix}> {d.name},[{bbox}],conf:{conf:.2f}, detail:{detail}, fn#:{fn}")


def print_detections(detections, frame_number = None, pre=None, post=None,labels=None):
    
    if pre          : print(pre)
    if frame_number : print(f">>>>>>>>>>>>>>>>>>>>> Frame {frame_number} <<<<<<<<<<<<<<<<<<<<<")

    for ix, d in enumerate(detections):
        
        if labels and d.name not in labels:
            # print(f'skipping {d.name} not in {labels}')
            continue # filter detections
        print_detection(d, index = ix )

    if post : print(post)


frames_to_debug = None # [1,2,3,4,5] 

def annotate_frame_text(frame, text, position, color=None, font_scale = 0.75, thickness = 1):
    if not color:
        color = (0, 255, 0)  # Text color in BGR (green)
    
    # Ensure frame is numpy array
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a numpy array")
    
    # Ensure text is string
    text = str(text)
    
    # Ensure position is tuple of integers
    position = tuple(map(int, position))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(
        frame, 
        text, 
        position, 
        font, 
        font_scale, 
        color, 
        thickness, 
        cv2.LINE_AA
    )
    
    return frame

def annotate_frame_pose_keypoints( frame, persons, kpt_color = None, edge_color = None, min_conf = 0.5, skeleton = True  ):

    # initialize annotator for plotting masks
    annotator = frameAnnotator(frame, line_width=2)
    frame_w = int(frame.shape[1])
    frame_h = int(frame.shape[0])
    
    if not kpt_color : kpt_color  = (  0, 255,   0)
    if not edge_color: edge_color = (255, 255, 255)

    try:
        for person_keypoints in persons:
            
            circles = {}
            
            for keypoint in person_keypoints:
                ix, name, x, y, conf = keypoint
                if conf > min_conf:  
                    cv2.circle(frame, (int(x), int(y)), radius=5, color=kpt_color, thickness=-1)
                    circles[ix] = (int(x),int(y))

            if skeleton:
                for from_ix, to_ix in COCO_SKELETON:
                    if from_ix in circles and to_ix in circles:

                        from_point = circles[ from_ix ]
                        to_point   = circles[ to_ix   ]

                        cv2.line(frame, from_point, to_point, color=edge_color, thickness = 2)

    except Exception as e:
        logging.error(f'annotate_frame_pose_keypoints - Error : {str(e)}')
        logging.info (f'annotate_frame_pose_key - persons {persons}')

    return frame

def hex_to_bgr(hex_color):
    '''
    Convert hex color #RRGGBB to OpenCV BGR tuple.
    '''
    hex_color = hex_color.lstrip('#')  # Remove '#' if present
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
    return (b, g, r)  # Convert to BGR

def color_map_init(cm_path):

    '''
    cm_path holds something like this:

    {
        "__comment__" : "Number keys are not valid-json, track_ids and frame numbers are strings..",
        "colors": {               
            "default": "#7F7F7F",
            "white": "#FFFFFF",
            "gray": "#7F7F7F",
            "warm_yellow": "#FFCC00",
            "orange_red": "#FF4500"
        },
        "tracks": {},
        "frames": {
                "3":  { "4": "orange_red" },
                "10": { "2": "warm_yellow" },
                "23": { "2": "default" },
                "36": { "4": "warm_yellow" },
                "40": { "4": "default" }
            }
    }

    '''

    color_map = None

    try:
        with open(cm_path, "r") as f:
            color_map = json.load(f)
    except FileNotFoundError as e:
        logging.info(f'No color map provided ({cm_path})- using defaults')
    except Exception as e:
        logging.error(f'color_map_init  - {str(e)}')
        color_map = None

    if color_map:
        # Convert hex colors to cv2 topples - not supported in json..
        for color_name, hex_color in color_map['colors'].items():
            color = hex_to_bgr(hex_color)
            color_map['colors'][color_name] = color

        # todo: convert frame dict from string key version to int key version
        # This would save conversion of track_id and frame numbers to strings

        logging.info(f'color_map {cm_path} loaded')

    return color_map


def color_map_update(color_map, frame_number):
    
    if color_map:
        frame = str(frame_number) # json does not support numeric keys..
        if frame in color_map['frames']:
            frame_track_id_colors = color_map['frames'][frame]
            for track_id, color_name in frame_track_id_colors.items():
                if color_name in  color_map['colors']:
                    color = color_map['colors'][color_name]
                else:
                    color = color_map['colors']['default']
                    logging.error(f'Could not find {color_name} in map! Using default color {color}')

                color_map['tracks'][track_id] = color
        
    return color_map


def annotate_frame(frame, detections, label = None, colors_map = None):
    
    # initialize annotator for plotting masks
    annotator = frameAnnotator(frame, line_width=2)

    for idx, detection in enumerate(detections):

        # Get the class name based on the class ID
        try:
            class_label = detection.name or "Unknown"
        except Exception as e:
            print(str(e))
            class_label = "Unknown"

        track_id = detection.track_id if detection.track_id else 0
        detail   = detection.detail

        detection_label = f'{class_label}({detection.conf:.2f}) '
        if detail   : detection_label = detection_label + detail + ' '
        if track_id : detection_label = detection_label + str(track_id)

        # Generate a color based on the track_id
        if colors_map :
            track_id_str = str(track_id) # json does not support numeric keys..
            if track_id_str in colors_map['tracks']:
                color = colors_map['tracks'][track_id_str]
                color = colors_map['colors']['default']
        else:
            color = colors(track_id, True)
    
        if detection.mask is not None: # has mask? draw mask
            annotator.seg_bbox(mask=detection.mask, mask_color=color, label=detection_label, txt_color=annotator.get_txt_color(color))
        else: # no mask - do box 
            annotator.box_label(detection.bbox, detection_label, color=color)

            if detection.inspect:
                annotator.centered_text(detection.bbox, ">> Inspect <<",box_style=True)

    if label:
        # Define the position and text for the frame number
        position = (10, 50)  # (x, y) position on the frame
        # Draw the text on the frame
        annotator.text(position, label,txt_color=(0,0,0),box_style=True)

    # annotation is done in place not by value, but still we return frame
    return frame

def build_name( name_parts, base_name = True ): 
        
    name_parts = [str(part) for part in name_parts if part ]
    if base_name:
        name_parts = [ os.path.basename(part)    for part in name_parts if part ]
        name_parts = [ os.path.splitext(part)[0] for part in name_parts if part ]

    name = '_'.join([str(part) for part in name_parts])

    return name

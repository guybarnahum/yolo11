import logging
import cv2
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

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
    Yolo sometimes confuses cars with bicycle, motorcycle, etc 
    '''
    cls_id_car_types = ["bicycle","motorcycle","bus","train","truck","boat","van"]
    cls_id_car       = None
    cls_id_car_list  = []
    
    logging.info( f"model classes: {cls_dict}")

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
            
            conf   = cls_results.boxes.conf[jdx].cpu().item()  # Confidence score

            # skip detections with low conf
            if min_conf and min_conf > conf:
                continue

            x1, y1, x2, y2 = cls_results.boxes.xyxy[jdx].cpu().tolist()
            
            if offset_x:
                x1 = x1 + offset_x
                x2 = x2 + offset_x
            
            if offset_y:
                y1 = y1 + offset_y
                y2 = y2 + offset_y

            bbox   = [x1, y1, x2, y2]
            
            area   = (x2 - x1) * (y2 - y1)
            if area < 0 : area = -area

            cls_id = int(cls_results.boxes.cls[jdx].cpu().item())  # Class ID
            cls_id = map_cls_id(cls_id) if map_cls_id else cls_id

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
            detection.area = area
            detection.frame_number = frame_number or -1
            detection.inspect = should_inspect(detection) if should_inspect else False
            detection.mask = mask
            detection.guid = None
            detection.track_id = track_id

            detections.append( detection )

    detections = non_max_suppression(detections)
    return detections


def print_detections(detections, frame_number = None):
    
    if frame_number:
        print(f">>>>>>>>>>>>>>>>>>>>> Frame {frame_number} <<<<<<<<<<<<<<<<<<<<<")

    for ix, d in enumerate(detections):
        conf = round(d.conf,2)
        bbox = [round(num, 2) for num in d.bbox]
        if d.track_id:
            print(f"{ix}> {d.track_id},{d.name},{bbox},{conf}")
        else:
            print(f"{ix}> {d.name},{bbox},{conf}")


frames_to_debug = None # [1,2,3,4,5] 

def annotate_frame(frame, detections, label = None):
    
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
        guid = f"{detection.guid} {track_id}" if detection.guid else track_id if track_id else None
        detection_label = f"{class_label} {guid} {detection.conf:.2f}" if guid else f"{class_label} - {detection.conf:.2f}" 

        # Generate a color based on the track_id
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

    return frame

def build_name( name_parts, base_name = True ): 
        
    name_parts = [str(part) for part in name_parts if part ]
    if base_name:
        name_parts = [ os.path.basename(part)    for part in name_parts if part ]
        name_parts = [ os.path.splitext(part)[0] for part in name_parts if part ]

    name = '_'.join([str(part) for part in name_parts])

    return name

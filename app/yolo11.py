import logging
from features.car.inspect    import should_inspect as should_inspect_car
from features.person.inspect import should_inspect as should_inspect_person

from trackers.deepsort.tracker import track as deepsort_track
from utils import print_detections, flatten_results

frames_to_debug = None # [1,2,3,4,5] 

def should_inspect( detection ):
    
    if detection.name == 'car':
        return should_inspect_car(detection)
    elif detection.name == 'person':
        return should_inspect_person(detection)

    return False


def process_one_frame( frame, detect_model, tile_model, tracker, tile, conf, frame_number = None, device='cpu' ):

    global frames_to_debug

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
                                        verbose=False,
                                        device=device
                                    )

        detections = flatten_results(   results, 
                                        min_conf=conf, 
                                        frame_number=frame_number,
                                        should_inspect=should_inspect,
                                        map_cls_id=detect_model.map_cls_id
                                    )

        detections = deepsort_track(detections, frame)

    else:
        try:
            results = detect_model.track( frame, 
                                          classes=class_codes,
                                          conf=conf,
                                          persist=True,
                                          verbose=False,
                                          tracker=tracker,
                                          device=device
                                        )

            detections = flatten_results(   results, 
                                            min_conf=conf, 
                                            frame_number = frame_number,
                                            should_inspect=should_inspect,
                                            map_cls_id=detect_model.map_cls_id
                                        )

        except Exception as e:
            logging.error(f"model.track error: {str(e)}")
            raise # not recoverable exception!

    # Debug specific frames
    if frames_to_debug and frame_number in frames_to_debug:
        print_detections(detections,frame_number)

    return frame, detections

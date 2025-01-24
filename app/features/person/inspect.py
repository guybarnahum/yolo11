import cv2
import logging

from types import SimpleNamespace
from utils import cuda_device, setup_model, print_detections
from utils import annotate_frame, annotate_frame_pose_keypoints, coco_pose_keypoint_name

topK = 1
face_detector = cv2.FaceDetectorYN.create( './models/face_detection_yunet_2023mar.onnx',  # Path to ONNX model
                                                '',                     # No configuration file
                                                input_size=(320, 320),  # Input size for the model
                                                top_k=topK              # Number of top faces to return
                                            )


pose_detector,_ = setup_model('./models/yolo11n-pose.pt')

def person_face( pr_frame, min_score = 0.7 ):
    
    w = int(pr_frame.shape[1])
    h = int(pr_frame.shape[0])

    # Set input size
    face_detector.setInputSize((w, h))

    # Getting detections
    _ , results = face_detector.detect(pr_frame)
    faces = []

    # if results and isinstance(results[1], list ) : # model generates (1,None) results..
    if results is not None:
        for result in results:
            if result is not None:
                try:
                    '''
                    results format
                    ( result_num,
                        [   # <--------------------------------------- results
                            [   
                                top, left, width, height,

                                left_eye_x    , left_eye_y    ,
                                right_eye_x   , right_eye_y   ,
                                nose_x        , nose_y        ,
                                left_mouth_x  , left_mouth_y  ,
                                right_mouth_x , right_mouth_y ,
                            ],
                            ...
                        ]
                    )
                    '''
                    x1, y1, width, height = result[:4]
                    x2 = x1 + width
                    y2 = y1 + height

                    # Extract keypoints
                    left_eye    = ( result[ 4], result[ 5])
                    right_eye   = ( result[ 6], result[ 7])
                    nose        = ( result[ 8], result[ 9])
                    left_mouth  = ( result[10], result[11])
                    right_mouth = ( result[12], result[13])

                    # Extract confidence score
                    conf = result[14]

                    faces.append( [[x1, y1, x2, y2], conf] )
                
                except Exception as e:
                    logging.error( f'person_face Error: {str(e)}' )
                    # continue ...
    
    return faces


def should_inspect(detection):
    '''
    Criteria for inspection 
    '''
    return detection.area > 120 * 120 # people are tall and thin most cases


def face_to_detection( face, offset_x = 0, offset_y = 0, frame_number = None ):

    face_x1, face_y1, face_x2, face_y2 = face[0]
    conf = face[1]

    area = ( face_x2 - face_x1 ) * ( face_y2 - face_y1 )
    if area < 0 : area = - area

    detection = SimpleNamespace()

    detection.bbox = [offset_x+face_x1, offset_y+face_y1, offset_x+face_x2, offset_y+face_y2]
    detection.conf = conf
    detection.cls_id = 0
    detection.name = 'face'
    detection.area = area
    detection.frame_number =  frame_number or -1
    detection.inspect = False
    detection.mask = None
    detection.guid = None
    detection.track_id = None

    return detection


def inspect_pose(person_detection, person_frame, video_path, offset_x = 0, offset_y = 0):
    
    device = cuda_device()
    if person_detection.name != 'person':
        logging.warning( f'features.person.inspect called for {person_detection}')
        return

    results = pose_detector.predict( source=person_frame,
                                     verbose=False,
                                     device=device 
                                    )
    persons = []
    try:
        if results:
            for cls_id, result in enumerate(results):
                
                kpts_sets   = result.keypoints
                confs_sets  = kpts_sets.conf.tolist()
                kpts_xy_sets= kpts_sets.xy.tolist()
                
                for person_id, (kpts_xy,confs) in enumerate(zip(kpts_xy_sets,confs_sets)):
                    
                    person = []

                    for index, (kpt_xy, conf) in enumerate(zip(kpts_xy,confs)): 
                        name = coco_pose_keypoint_name(index)
                        x, y = kpt_xy 
                        if conf > 0.7:  # Filter out low-confidence keypoints
                            person.append([index,name,x + offset_x,y + offset_y,conf])
                    
                persons.append(person)
                
            return persons
    
    except AttributeError as e: 
        logging.warning(f'inspect_pose :{str(e)}')

    except Exception as e:
        logging.error(f'inspect_pose Error: {str(e)}')
    
    return persons


def inspect(person_detection, frame, video_path):
    
    device = cuda_device() # Unused at this time..

    if person_detection.name != 'person':
        logging.warning( f'features.person.inspect called for {person_detection}')
        return

    x1, y1, x2, y2 = person_detection.bbox
    pr_frame = frame[ int(y1): int(y2), int(x1): int(x2), :]

    faces = person_face(pr_frame)
    detections = []

    if faces :
        for face in faces:
            detection = face_to_detection( face, offset_x = x1, offset_y = y1, frame_number = person_detection.frame_number )
            detections.append( detection )     
    
    persons = inspect_pose(person_detection, pr_frame, video_path, offset_x = x1, offset_y = y1)
    frame = annotate_frame_pose_keypoints( frame, persons )

    if len(detections):
        print_detections( detections, frame_number = person_detection.frame_number )
        annotate_frame(frame, detections)

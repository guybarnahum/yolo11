import logging
import json
import os
from pprint import pprint
from types import SimpleNamespace
from random import randint

def gt_load_mot( gt_path ):
    '''
    MOT 1.1 has text lines with:
        <frame>, <id>, <x>, <y>, <width>, <height>, <confidence>, <class_id>, <visibility>

    We add a first line with json header, with labels info:
    
   {'signature':'mot1.1','noise':10,'comment':'cvat export', 'labels':'person,face,bicycle,car,license_plate'}

    Example:
    
    {'signature':'mot1.1','noise':10,'comment':'cvat export', 'labels':'person,face,bicycle,car,license_plate'}
    1,1,593.57,467.34,229.64,462.49000000000007,1,3,1.0 
    1,2,382.28,134.2,696.24,512.6600000000001,1,4,1.0
    1,3,109.69,93.43,343.13,363.82,1,4,1.0
    1,4,241.27,124.1,138.4,423.91999999999996,1,1,1.0
    .
    .
    .
    '''

    # Do we have groud truth?
    if not os.path.isfile( gt_path ):
        logging.info(f'No mot 1.1 ground truth file found at {gt_path}')
        return None

    frame_dict = {}

    with open(gt_path, 'r') as f:

        # Read the signature line and labels
        try:
            header_str = f.readline().strip()
            header = json.loads(header_str)

        except Exception as e:
            logging.error(f'gt_load_mot - {str(e)}')
            header = {'header' : None }

        # Check signature
        signature = header['signature'] if 'signature' in header else None
        if signature != 'mot1.1' :
            logging.error(f'Failed to load mot1.1 from {gt_path}')
            return None

        comment =    header['comment'] if 'comment' in header else ''
        noise   = int(header['noise']) if 'noise'   in header else 0

        logging.info(f'gt_load_mot - loaded mot file : {signature}/{comment}')
        if noise:
            logging.info(f'gt_load_mot - noise : +/- {noise}')
        # Load labels for class_ids
        try:
            labels = header['labels'].split(',')
        except Exception as e:
            logging.error(f'gt_load_mot - Invalid labels in header {header} : {str(e)}')
            return None

        # Process MOT 1.1 data lines
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 9:
                logging.warning(f'gt_load_mot - malformed line skipped : {line}')
                continue  # Skip malformed lines

            frame_number= int(parts[0]) - 1
            track_id    = int(parts[1])
            x, y, width, height = map(float, parts[2:6])

            if noise:
                x += randint(-noise,noise)
                y += randint(-noise,noise)

            conf        = float(parts[6])
            class_id    = int(parts[7])
            visibility  = float(parts[8])

            # Replace class_id with label name
            name = labels[class_id - 1] if 0 < class_id <= len(labels) else 'unknown'
            if name == 'unknown':
                logging.warning(f'gt_load_mot - unknown label for class_id : {class_id}')


            groud_truth = SimpleNamespace()
            groud_truth.track_id = track_id
            groud_truth.bbox = [ x, y, x + width, y + height ]
            groud_truth.conf = conf
            groud_truth.name = name
            groud_truth.visibility = visibility
            groud_truth.frame_number = frame_number

            # Add detection to the corresponding frame
            if frame_number not in frame_dict: 
                frame_dict[frame_number] = []

            frame_dict[frame_number].append(groud_truth)

    # Pack into gt object
    gt = {
        "labels" : labels,
        "frames" : frame_dict
    }

    return gt

def gt_get_frame_detections( gt, frame_number, labels = [] ):

    if not gt : 
        # No groudthrough ..
        logging.warning('gt_get_frame_detections - no groundtruth frame set')
        return None

    if frame_number not in gt['frames']:
        logging.warning(f'gt_get_frame_detections - missing frame {frame_number} from groundtruth frame set')
        return None

    frame_detections = gt['frames'][frame_number]

    if labels == []:
        return frame_detections

    # We need to filter frame detections by labels
    filtered_frame_detections = []

    for d in frame_detections:
        if d.name in labels:
            filtered_frame_detections.append(d)
    
    return filtered_frame_detections


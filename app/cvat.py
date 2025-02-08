import logging
import json 
import os
import shutil

from hashlib import md5
from cv2 import imencode

from utils import print_detection

def cvat_init(name, video_path, width, height,num_frames): 
    # Initial structure for the CVAT JSON
    
    video_name = os.path.basename(video_path)

    return {
        
        "video_path": video_path,

        # track_ids dict has track_id as key and index into tracks array for later access
        # notice we don't record the track_id anywhere..

        "track_ids"          : {}, # map global track_ids to tracks array indexes..
        "lost_track_indexes" : [], # array of indexs for tracks we saw and lost ...

        "manifest_header": [
            '{"version":"1.1"}',
            '{"type":"video"}',
            '{"properties":{"name":"'+video_name+'","resolution":['+str(width)+','+str(height)+'],"length":'+str(num_frames)+'}}'
        ],
        
        "manifest_frames": [],
        
        "label_names" : [ "person", "car", "face", "bicycle", "license_plate" ],

        "task": {
                "version": "1.0",
                "name"  : f"{name}_cvat",
                "description": "This task involves annotating objects in a {video_name}," 
                               "make sure to upload and assicate the video with the task",
                "mode"  : "interpolation",
                "status": "annotation",
                "subset": "",
                "bug_tracker": "",                
                "data": {
                    "chunk_size": 36,
                    "image_quality": 70,
                    "start_frame": 0,
                    "stop_frame" : num_frames - 1,
                    "storage_method": "cache",
                    "storage": "local",
                    "sorting_method": "lexicographical",
                    "chunk_type": "imageset",
                    "deleted_frames": []
                },
                "labels": [
                    {
                        "name": "person",
                        "attributes": [],
                        "color": "#ff5733", # Bright Orange-Red
                        "type": "rectangle",
                        "sublabels": []
                    },
                    {
                        "name": "face",
                        "attributes": [],
                        "color": "#ffc300", # Warm Yellow
                        "type": "rectangle",
                        "sublabels": []
                    },
                    {
                        "name": "bicycle",
                        "attributes": [],
                        "color": "#33ff57", # Vivid Green
                        "type": "rectangle",
                        "sublabels": []
                    },
                    {
                        "name": "car",
                        "attributes": [
                            {
                                "name": "yaw",
                                "input_type": "number",
                                "mutable": True,
                                "values": [],
                                "default_value" : "-1"
                            },
                            {
                                "name": "yaw_conf",
                                "input_type": "number",
                                "mutable": True,
                                "values": [],
                                "default_value" : "0.0"
                            },
                            {
                                "name": "license_plate",
                                "input_type": "text",
                                "mutable": True,
                                "values": [],
                                "default_value" : ""
                            },
                            {
                                "name": "license_plate_conf",
                                "input_type": "number",
                                "mutable": True,
                                "values": [],
                                "default_value" : "0.0"
                            }
                        ],
                        "color": "#3380ff",  # Bright Blue
                        "type": "rectangle",
                        "sublabels": []
                    },
                    {
                        "name": "license_plate",
                        "attributes": [
                            {
                                "name": "ocr_text",
                                "input_type": "text",
                                "mutable": True,
                                "values": [],
                                "default_value" : ""
                            },
                            {
                                "name": "ocr_text_conf",
                                "input_type": "number",
                                "mutable": True,
                                "values": [],
                                "default_value" : "0.0"
                            }  
                        ],
                        "color": "#a833ff", # Purple
                        "type": "rectangle",
                        "sublabels": []
                    },
                ],
                "jobs": [
                    {
                        "start_frame": 0,
                        "stop_frame": num_frames - 1,
                        "frames": [],
                        "status": "annotation",
                        "type": "annotation"
                    }
                ]
        },

        "annotations": [{
                        "version": "1.0",
                        "tags"  : [],
                        "shapes": [],
                        "tracks": []
                   }]
    }


def cvat_add_frame_to_manifest( cvat_json, frame, frame_number, fps, force = True ):

    # Calculate pts (presentation timestamp) interval
    pts_interval = int(1000 / fps)  # milliseconds per frame
    
    if force or frame_number % fps == 0:
        # Calculate checksum
        _, buffer = imencode('.jpg', frame)
        checksum = md5(buffer).hexdigest()
        pts = frame_number * pts_interval
        frame_info = '{"number":' + str(frame_number) +',"pts":' + str(pts) +',"checksum":"'+checksum+'"}'

        cvat_json['manifest_frames'].append(frame_info)


def cvat_add_frame( cvat_json, detections, frame_number) :
    """
    Convert frame annotations to CVAT JSON format and add to the annotations.
    
    Args:
        frame_number: The frame index (0-based).
        detections: List of detections for the current frame. Each detection is a dictionary with 'label', 'bbox', 'track_id'.
        cvat_json: The current CVAT JSON object that stores the annotations.
    """

    track_index_in_frame = set()

    for ix, detection in enumerate(detections):
        
        label       = detection.name

        # CVAT is strict in terms of which labels are included ..
        if label not in cvat_json[ "label_names" ]:
            logging.warning(f'CVAT unsupported label {label} - skipped ...')
            continue

        bbox        = detection.bbox  # [xmin, ymin, xmax, ymax]
        track_id    = detection.track_id
        points      = [bbox[0], bbox[1], bbox[2], bbox[3]]
        attributes  = detection.attributes
        
        debug_detection = False and label in ['face', 'license_plate']

        if debug_detection:
            print_detection( detection , index = ix)

        if track_id :

            shape = {
                "type"  : "rectangle",
                "occluded" : False,
                "outside"  : False,
                "z_order": 0,
                "rotation": 0.0,
                "points": points,
                "frame" : frame_number,
                "attributes": attributes,
            }

            track_id = f"{label}:{track_id}"

            # New tracked object?
            if track_id not in cvat_json['track_ids']:
                
                # track_ids[track_id] is the index for the new track record
                track_index = len(cvat_json['annotations'][0]['tracks'])
                cvat_json['track_ids'][track_id] = track_index
                cvat_json['annotations'][0]['tracks'].append({  "label": label,
                                                                "frame": frame_number, # first frame?
                                                                "group": 0,
                                                                "source": "auto",
                                                                "attributes": [], # <-- Non mutable attaributes go here
                                                                "elements": [], 
                                                                "shapes": [shape]
                                                        })
                if debug_detection:
                    logging.info(f'cvat_add_frame - inserting new track_id : {track_id} shape to tracks array index : {track_index}')
                
            else: # tarck already exists - add shape
                track_index = cvat_json['track_ids'][track_id]
                cvat_json['annotations'][0]['tracks'][track_index]['shapes'].append(shape)

                if debug_detection:
                    logging.info(f'cvat_add_frame - inserting track_id : {track_id} shape to tracks array index : {track_index}')

            '''
                Update tracks seen this frame and update lost_track_indexes 
                We do for both cases even if only needed for existing tracks not new ones..
            '''

            track_index_in_frame.add( track_index ) # Add track to existing tracks in frame

            if track_index in cvat_json['lost_track_indexes']:
                cvat_json['lost_track_indexes'].remove(track_index) # track is not lost anymore..
                logging.debug(f"lost track_index {track_index} found again! - {cvat_json['lost_track_indexes']}")
            

        else: # a detection without track_id
            shape = {
                "label" : label,
                "source": "auto",
                "type"  : "rectangle",
                "occluded" : False,
                "outside"  : False,
                "z_order": 0,
                "rotation": 0.0,
                "points": points,
                "frame" : frame_number,
                "attributes": attributes,
            }

            if debug_detection:
                logging.info(f'cvat_add_frame - inserting no-track-id {label} shape to shapes array')

            cvat_json['annotations'][0]['shapes'].append(shape)
    

    # Mark track missing with a hidden / outside shape 
    shape_outside ={
        "type"       : "rectangle",
        "frame"      : frame_number,
        "occluded"   : False,
        "outside"    : True,
        "keyframe"   : True,
        "z_order"    : 0,
        "rotation"   : 0.0,
        "points"     : [],
        "attributes" : []
    }

    for track_index, track in enumerate(cvat_json['annotations'][0]['tracks']):
        
        '''
            Only newly lost tracks deserve a shape_outside marker
            `newly lost` is `missing in this frame` AND `is not already lost`
            We rest these in track logic above for found track_ids
        '''
        if  track_index not in track_index_in_frame and track_index not in cvat_json['lost_track_indexes']:

            # Not sure why CVAT needs this, but it insists on points for outside rectangle..
            # Maybe to clear the screen from it?
            shape_outside['points'] = track['shapes'][-1]['points']

            track['shapes'].append(shape_outside) # track is newly lost track - mark it!
            cvat_json['lost_track_indexes'].append(track_index) 

            logging.debug(f"new lost track_index: {track_index}")

    logging.debug(f"frame tracks set: {track_index_in_frame}")
    logging.debug(f"lost_tracks array: {cvat_json['lost_track_indexes']}")

    # Update the task size (total number of frames)
    # cvat_json['task']["data"]["stop_frame"] += 1

    return cvat_json


def cvat_save_json( data, base_dir, file_name ):
    
    json_path = os.path.join(base_dir, file_name)

    try:
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
    
        # Write the task data to task.json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logging.debug(f'Saved data into {json_path}')

    except Exception as e:
        logging.error(f'cvat_save_json : {str(e)}')

    return json_path


def cvat_save(cvat_json, base_dir):
    '''
    Save the generated CVAT JSON to a file.
    '''
    # print(cvat_json)

    task_data = cvat_json['task']
    task_name = task_data['name']
    video_path = cvat_json['video_path'] 

    # task.js
    base_dir = os.path.join(base_dir, task_name )
    cvat_save_json( task_data, base_dir, 'task.json' )

    # annotations.js
    annotations_data = cvat_json['annotations']
    cvat_save_json( annotations_data, base_dir, 'annotations.json' )

    # manifest.jsonl
    data_base_dir = os.path.join(base_dir     ,'data')
    manifest_path = os.path.join(data_base_dir,'manifest.jsonl')

    os.makedirs(data_base_dir, exist_ok=True)

    with open(manifest_path, 'w') as f:
        manifest_header = cvat_json['manifest_header']
        f.write('\n'.join(manifest_header) + '\n')

        manifest_frames = cvat_json['manifest_frames']
        f.write('\n'.join(manifest_frames) + '\n')

    # Zip it
    shutil.copy(video_path,data_base_dir)
    archive_path = shutil.make_archive(base_dir, "zip", base_dir)
    
    # remove data_base_dir (!)
    # shutil.rmtree(data_base_dir) 
    shutil.rmtree(base_dir) # <-- disbale to inspect archive in IDE

    return archive_path


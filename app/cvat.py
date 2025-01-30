import logging
import json 

labels_attributes_spec_ids = {
    'yaw'                 : 10,
    'yaw_conf'            : 11,

    'license_plate'       : 20,
    'license_plate_conf'  : 21,

    'ocr_text'            : 30,
    'ocr_text_conf'       : 31,
}

def spec_id( attribute ):
    
    if attribute in labels_attributes_spec_ids:
        return labels_attributes_spec_ids[ attribute ]

    return None

def cvat_init(name,video_path):
    # Initial structure for the CVAT JSON
    return {
        "version": "1.1",
        "tasks": [
            {
                "id": 1,
                "name": f"{name}_task",
                "size": 0,  # will be updated later with total number of frames
                "mode": "interpolation",
                "overlap": 0,
                "bug_tracker": "",
                "labels": [
                    {"name": "person", "attributes": []},
                    {"name": "face", "attributes": []},
                    
                    {   "name": "car", 
                        "attributes":   [ 
                            {
                                "spec_id": spec_id("yaw"),
                                "name": "yaw",
                                "input_type": "number",
                                "mutable": True
                            },
                            {
                                "spec_id": spec_id("yaw_conf"),
                                "name": "yaw_conf",
                                "input_type": "number",
                                "mutable": True
                            },
                            {
                                "spec_id": spec_id("license_plate"),
                                "name": "license_plate",
                                "input_type": "text",
                                "mutable": True
                            },
                            {
                                "spec_id": spec_id("license_plate_conf"),
                                "name": "license_plate_conf",
                                "input_type": "number",
                                "mutable": True
                            }
                        ]
                    },

                    {   "name": "license_plate", 
                        "attributes": [
                            {
                                "spec_id": spec_id("ocr_text"),
                                "name": "ocr_text",
                                "input_type": "text",
                                "mutable": True
                            },
                            {
                                "spec_id": spec_id("ocr_text_conf"),
                                "name": "ocr_text_conf",
                                "input_type": "number",
                                "mutable": True
                            }                            
                        ]
                    },
                ],
                "image_quality": 100,
                "annotations": [],
                "video_path": video_path,
                "description": "This task involves annotating objects in a {video_path}, make sure to upload and assicate the video with the task"
            }
        ]
    }

def cvat_add_frame(frame_number, detections, cvat_json):
    """
    Convert frame annotations to CVAT JSON format and add to the annotations.
    
    Args:
        frame_number: The frame index (0-based).
        detections: List of detections for the current frame. Each detection is a dictionary with 'label', 'bbox', 'track_id'.
        cvat_json: The current CVAT JSON object that stores the annotations.
    """
    objects = []

    for ix, detection in enumerate(detections):
        id          = f"{frame_number}.{ix}"
        label       = detection.name
        bbox        = detection.bbox  # [xmin, ymin, xmax, ymax]
        track_id    = detection.track_id
        points      = [bbox[0], bbox[1], bbox[2], bbox[3]]
        attributes  = detection.attributes

        objects.append({
            "id": id,
            "label": label,
            "points": points,
            "attributes": attributes,  
            "track_id": track_id # Can be None
        })

    # Add frame annotations to the task
    task = cvat_json['tasks'][0]
    
    task["annotations"].append({
        "frame": frame_number,
        "objects": objects
    })

    # Update the task size (total number of frames)
    task["size"] = task["size"] + 1

    return cvat_json

def cvat_save(cvat_json, file_path):
    '''
    Save the generated CVAT JSON to a file.
    '''
    with open(file_path, 'w') as f:
        json.dump(cvat_json, f, indent=4)

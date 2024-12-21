from deep_sort_realtime.deepsort_tracker import DeepSort, Detection

# Initialize DeepSort tracker
deepsort_tracker = DeepSort(max_age=150) # 150 frames or 5 sec memory

def yolo_to_ltwh(xyxy):
    """Convert [x1, y1, x2, y2] bounding box to [left, top, width, height] format."""
    x1, y1, x2, y2 = xyxy
    left = x1
    top = y1
    width = x2 - x1
    height = y2 - y1
    return [left, top, width, height]


def track(results, frame):
    # For each bounding box, create a Detection object
    detections = []
    others = []

    for idx, result in enumerate(results):
        for jdx in range(len(result.boxes.xyxy)):
            bbox_xyxy = result.boxes.xyxy[jdx].cpu().tolist()  # [x1, y1, x2, y2]
            confidence = result.boxes.conf[jdx].cpu().item()   # Confidence score
            class_id = int(result.boxes.cls[jdx].cpu().item()) # Class ID
            
            # Convert bbox to [left, top, width, height] format
            bbox_ltwh = yolo_to_ltwh(bbox_xyxy)

            # Create a Detection object and store the YOLO index in the 'others' field
            detection = [bbox_ltwh, confidence, class_id]
            detections.append(detection)
            others.append({'idx': idx, 'jdx':jdx})

    # Update the tracker with the detections
    tracks = deepsort_tracker.update_tracks(detections, frame=frame, others = others)
    
    # Pack tracker ids as yolo results lists
    track_ids = []
    for ix, result in enumerate(results):
        track_ids.append( [] )
        for jdx in range(len(result.boxes.xyxy)):
            track_ids[ix].append( 0 )
            
    for track in tracks:
        if track.is_confirmed():
            others   = track.get_det_supplementary()
            if others: 
                track_id = track.track_id  # Unique track ID

                #ltrb     = track.to_ltrb(orig=True, orig_strict=True)
                #cls_id   = track.get_det_class()
                #conf     = round(track.get_det_conf(),2)
                #print(f"Track {track_id} - others: {others} conf:{conf} cls_id:{cls_id} ltrb:{ltrb}")
           
                idx = others['idx']
                jdx = others['jdx']
                track_ids[idx][jdx] = track_id
                #print(f"({idx},{jdx}) <= {track_id}")

    #print(f"track_ids: {track_ids}")
    return track_ids
'''

yolo11-tracker  | Track 1 - others: {'idx': 0, 'jdx': 5} conf:0.6010926365852356 cls_id:0 ltrb:[     804.88       160.8      1018.7      725.87]
yolo11-tracker  | Track 2 - others: {'idx': 0, 'jdx': 0} conf:0.9082240462303162 cls_id:0 ltrb:[     125.24      4.5038      555.88      1013.3]
yolo11-tracker  | Track 3 - others: {'idx': 0, 'jdx': 3} conf:0.22563982009887695 cls_id:0 ltrb:[     747.51      151.26      897.88       725.6]
yolo11-tracker  | Track 4 - others: None conf:None cls_id:0 ltrb:None
yolo11-tracker  | Track 9 - others: {'idx': 0, 'jdx': 4} conf:0.8381849527359009 cls_id:0 ltrb:[     594.42      115.48      770.97       729.1]
yolo11-tracker  | Track 10 - others: {'idx': 0, 'jdx': 6} conf:0.5879455208778381 cls_id:0 ltrb:[     717.87      119.22      836.76      735.76]
yolo11-tracker  | Track 24 - others: {'idx': 0, 'jdx': 1} conf:0.8435826897621155 cls_id:0 ltrb:[     1491.3      117.66      1649.7      759.64]
yolo11-tracker  | Track 26 - others: None conf:None cls_id:0 ltrb:None
yolo11-tracker  | Track 27 - others: {'idx': 0, 'jdx': 2} conf:0.824719250202179 cls_id:0 ltrb:[    0.33862      4.9039      142.77      982.48]
yolo11-tracker  | Track 32 - others: None conf:None cls_id:0 ltrb:None
yolo11-tracker  | Track 33 - others: None conf:None cls_id:0 ltrb:None

'''

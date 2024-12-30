from deep_sort_realtime.deepsort_tracker import DeepSort, Detection
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('deepsort_tracker')

class InstrumentedDeepSort(DeepSort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        
    def update_tracks(self, detections, frame=None, others=None):
        self.frame_count += 1
        
        # Log incoming detections
        for det_idx, det in enumerate(detections):
            bbox, conf, cls_id = det
            logger.debug(f"Frame {self.frame_count} Detection {det_idx}: "
                        f"bbox={bbox}, conf={conf:.2f}, class={cls_id}")

        # Get current tracks before update
        prev_tracks = {t.track_id: (t.to_ltrb(), t.time_since_update) 
                      for t in self.tracker.tracks}

        # Run tracker update
        tracks = super().update_tracks(detections, frame, others)

        # Analyze tracking results
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                curr_bbox = track.to_ltrb(orig=True)
                conf = track.get_det_conf()
                cls_id = track.get_det_class()
                
                logger.debug(
                    f"Track {track_id} - conf:{conf:.2f} cls:{cls_id} bbox:{curr_bbox}")
                
                # Check if track existed before
                if track_id in prev_tracks:
                    prev_bbox, time_since_update = prev_tracks[track_id]
                    
                    # Calculate changes
                    size_change = self._calc_size_change(prev_bbox, curr_bbox)
                    displacement = self._calc_displacement(prev_bbox, curr_bbox)
                    
                    # Log significant changes
                    if size_change > 0.5:
                        logger.warning(
                            f"Track {track_id}: Large size change: {size_change*100:.1f}%")
                            
                    if displacement > 100:
                        logger.warning(
                            f"Track {track_id}: Large displacement: {displacement:.1f}px")
                            
                    if time_since_update > 5:
                        logger.warning(
                            f"Track {track_id}: Lost for {time_since_update} frames")
                            
            else:
                logger.debug(f"Track {track.track_id} - Unconfirmed")

        return tracks
        
    def _calc_size_change(self, bbox1, bbox2):
        """Calculate relative size change between two bounding boxes"""
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        size1, size2 = w1 * h1, w2 * h2
        return abs(size2 - size1) / size1 if size1 > 0 else float('inf')
    
    def _calc_displacement(self, bbox1, bbox2):
        """Calculate center point displacement between two bounding boxes"""
        c1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
        c2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
        return ((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)**0.5

def yolo_to_ltwh(xyxy):
    """Convert [x1, y1, x2, y2] bounding box to [left, top, width, height] format."""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2-x1, y2-y1]

def track(results, frame):
    deepsort_tracker = InstrumentedDeepSort(
        max_age=150,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )
    
    # Process detections
    detections = []
    others = []
    
    for idx, result in enumerate(results):
        for jdx in range(len(result.boxes.xyxy)):
            bbox_xyxy = result.boxes.xyxy[jdx].cpu().tolist()
            confidence = result.boxes.conf[jdx].cpu().item()
            class_id = int(result.boxes.cls[jdx].cpu().item())
            
            bbox_ltwh = yolo_to_ltwh(bbox_xyxy)
            detection = [bbox_ltwh, confidence, class_id]
            detections.append(detection)
            others.append({'idx': idx, 'jdx':jdx})

    tracks = deepsort_tracker.update_tracks(detections, frame=frame, others=others)
    
    track_ids = []
    for ix, result in enumerate(results):
        track_ids.append([0] * len(result.boxes.xyxy))
            
    for track in tracks:
        if track.is_confirmed():
            others = track.get_det_supplementary()
            if others:
                track_id = track.track_id
                idx = others['idx']
                jdx = others['jdx']
                track_ids[idx][jdx] = track_id
                
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

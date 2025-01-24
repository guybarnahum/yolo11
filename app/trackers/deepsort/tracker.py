import logging
import os
from deep_sort_realtime.deepsort_tracker import DeepSort, Detection
from utils import cuda_device

# Workaround to https://github.com/levan92/deep_sort_realtime/issues/57
#
# Basically TorchReID_Embedder has: 
#   from torchreid.utils import FeatureExtractor
# which does not exists, it should be:
#   from torchreid.reid.utils import FeatureExtractor
#
import sys
from torchreid.reid import utils as torchreid_reid_utils
sys.modules["torchreid.utils"] = torchreid_reid_utils

#from deep_sort_realtime.enbedder import Clip_Embedder
#clip_embedder = Clip_Embedder(model_wts_path='./ViT-B-32.pt', gpu=False)

deepsort_tracker = None

def setup(embedder = None,embedder_wts=None):

    device = cuda_device()

    # Examples:          
    # embedder='clip_ViT-B/16',
    # embedder_wts='./trackers/deepsort/clip_ViT-B-16.pt' OR None
    # OR
    # embedder='torchreid',
    # embedder_wts='osnet_ain_x1_0_imagenet.pth' OR 'osnet_ain_x1_0_imagenet'

    global deepsort_tracker

    # Look for embedder_wts
    base_name = embedder_wts if embedder_wts else embedder.replace("/", "-")
    embedder_wts_paths = []
    embedder_wts_base_paths = ['','./models/embedders', '.']

    for base_path in embedder_wts_base_paths:
        embedder_wts_paths.append( os.path.join(base_path,base_name) )
        embedder_wts_paths.append( os.path.join(base_path,base_name +'.pt') )
        embedder_wts_paths.append( os.path.join(base_path,base_name +'.pth') )

    embedder_wts_paths.append( None ) # Mark the end of places to try

    for embedder_wts_path in embedder_wts_paths:
        
        if not embedder_wts_path:
            logging.warning(f'Could not locate embedder weights')
            embedder_wts = None
            break

        logging.debug(f"Looking for {embedder_wts_path}")

        if os.path.isfile(embedder_wts_path):
            embedder_wts = embedder_wts_path
            logging.debug(f'located embedder weights at {embedder_wts}')
            break

    logging.info( f"Using embedder_wts : {embedder_wts}")

    # Setup DeepSort tracker
    deepsort_tracker = DeepSort(max_age=900, # 900 frames or 30 sec memory
                                max_iou_distance=1.0,
                                nn_budget=200,
                                embedder=embedder,
                                embedder_wts=embedder_wts,
                                embedder_gpu= device != 'cpu'
                        )


def yolo_to_ltwh(xyxy):
    """Convert [x1, y1, x2, y2] bounding box to [left, top, width, height] format."""
    x1, y1, x2, y2 = xyxy
    left = x1
    top = y1
    width = x2 - x1
    height = y2 - y1
    return [left, top, width, height]


def track(detections, frame):
    
    global deepsort_tracker

    # Create a DeepSort detection object
    detections_ds = []
    others = []

    for idx, detection in enumerate(detections):

        detection.track_id = None
        bbox_ltwh = yolo_to_ltwh(detection.bbox)
        
        # Create a deepsort detection object and store the YOLO index in the 'others' field
        detection_ds = [bbox_ltwh, detection.conf, detection.cls_id]
        detections_ds.append(detection_ds)
        others.append({'idx': idx})

    # Update the tracker with the detections
    tracks = deepsort_tracker.update_tracks(detections_ds, frame=frame, others = others)
    
    # Update track_id into original detection
    for track in tracks:
        if track.is_confirmed():
            others = track.get_det_supplementary()
            if others: 
                track_id = track.track_id  # Unique track ID
                idx = others['idx']

                detections[idx].track_id = track_id

    return detections

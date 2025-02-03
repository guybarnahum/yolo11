import logging

'''
    Logging setLevel Hack!

    We have a problem of roug modules (paddles?) who set the root Logger to WARNING, overriding our level.
    Below, is a special setLevel that denies the change the level and blames those who try to set it.

'''
import traceback
import inspect # to blame culprit

root_logging_level = logging.INFO
_setLevel = logging.getLogger().setLevel 

def setLevel_locked(level):
    if level != root_logging_level:

        # Walk up the call stack to find the real culprit
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        module  = inspect.getmodule(caller_frame)
        culprit = module.__name__

        logging.warning(f"⚠️  Denied attempt of ({culprit}) to change the root logger level to {logging.getLevelName(level)}")
        return # do noting - logger is lock to changes

    _setLevel(level)


# Patch the root logger
logging.getLogger().setLevel = setLevel_locked

# Set Root Logger level - basicConfig still works
logging.basicConfig(level=root_logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

'''
    END of Logging Hack 
'''

import cv2
from dotenv import load_dotenv
import importlib

from fastapi import BackgroundTasks, FastAPI, Query
import fiftyone as fo

import os
from pathlib import Path
import sys
import time
from tqdm.auto import tqdm #select best for enviroment
from typing import Optional

from compress_video import compress_video_to_size, compress_video_to_bitrate, get_bitrate, calculate_bit_rate
from yolo11 import process_one_frame
from features.inspect import inspect

from trackers.deepsort.tracker import setup as deepsort_setup
from utils import annotate_frame, build_name, cuda_device, setup_model, print_detections
from cvat import cvat_init, cvat_add_frame, cvat_save

load_dotenv()
app = FastAPI()

def process_video(  model_path, process_one_frame_func, 
                    input_path, output_path, dataset_path=None,
                    tracker=None, embedder = None, embedder_wts = None,
                    tile=None, 
                    start_ms=None, end_ms=None, conf=None,
                    cvat=False
                ):
    """
    Main video processing process
    """
    device = cuda_device()

    if tracker == 'deepsort':
        deepsort_setup(embedder=embedder, embedder_wts=embedder_wts)

    embedder_name = embedder.replace('/','-') if embedder else None

    # Check if the dataset exists, otherwise create an empty video dataset
    dataset = None

    if dataset_path:
        dataset_type=fo.types.FiftyOneVideoLabelsDataset
        try:
            logging.info(f"Loading dataset from {dataset_path}")
            dataset = fo.Dataset.from_dir( dataset_dir=dataset_path, dataset_type=dataset_type)
            logging.info(f"Loading finished...")
        except Exception as e:
            logging.error(f"🚨 Exception: {str(e)}")
            dataset = None

        if not dataset:
            dataset = fo.Dataset()
            dataset.media_type = "video" 
            dataset.persistent = True
            # Define the schema for frame-level fields
            dataset.add_frame_field("detections", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
            logging.info(f"dataset created")

    model_label = build_name( [ 'model', model_path, tracker, tile, embedder_name ] )
    logging.info(f"model_label {model_label}")

    if dataset:
        sample = fo.Sample(filepath=input_path)
   
    # Video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Failed to open input video")
        return

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                           cv2.CAP_PROP_FRAME_HEIGHT, 
                                           cv2.CAP_PROP_FPS))
    frame_ms  = int(1000 / fps)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    # Load models from the config
    detect_model, tile_model = setup_model(model_path, tile, image_size=w, build_cls_map=True)
    logging.info(f"detect-model: {model_path}, tile: {tile}, conf: {conf} device: {device}")

    # Frame calculations
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(start_ms * fps / 1000) if start_ms else 0   
    end_frame    = int(end_ms * fps / 1000) if end_ms else (total_frames - 1)

    frames_to_process = end_frame - start_frame
    progress_bar = tqdm(total=frames_to_process)
    one_percent  = int(frames_to_process / 100) + 1

    logging.info(f"{frames_to_process} from [{start_frame}..{end_frame}]")
    logging.info(f"one_percent : {one_percent} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if cvat:
        input_stem = Path(input_path).stem
        cvat_json = cvat_init( name=input_stem,video_path=input_path,width=w, height=h, num_frames=total_frames)

    for frame_ix in range(start_frame, end_frame):
        ret, im0 = cap.read()
        if not ret:
            break

        im0, detections = process_one_frame_func(   im0, 
                                                    detect_model, tile_model, tracker, 
                                                    tile, conf, 
                                                    frame_number=frame_ix, 
                                                    device=device)
        
        # Enqueue detections that need further inspection
        features = []

        for detection in detections:
            if detection.inspect:
                detection_features = inspect( detection, frame=im0, video_path=input_path)
                if detection_features:
                    features.extend(detection_features)
                
        # print_detections( features, frame_number = frame_ix )
        detections.extend(features)
        
        # annotae the frame after inspection job
        label = f" {model_label} FN#{frame_ix} "
        annotate_frame(im0, detections, label=label)
        out.write(im0)

        if  cvat:
            cvat_json = cvat_add_frame(cvat_json, detections, im0, frame_ix, fps)

        if dataset:
            detections_list = []
            for detection in detections:
                # Convert to [top-left-x, top-left-y, width, height]
                 # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = detection.bbox  
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                
                detection_fo =  fo.Detection(
                        label=detection.name,
                        bounding_box=rel_box,
                        confidence=detection.conf,
                        track_id=detection.track_id 
                    )

                detection_fo.tags.append(model_label)

                detections_list.append( detection_fo )

            detections_obj = fo.Detections(detections=detections_list)    
            frame_obj = fo.Frame(detections=detections_obj)
      
            sample.frames[ frame_ix + 1 ] = frame_obj
        
        if frame_ix % one_percent == 0:
            print(".")

        progress_bar.update(1)
    
    if  cvat:
        # generate the cvat files inside output sub-direcotry
        output_base_dir = os.path.dirname(output_path)
        cvat_save(cvat_json, output_base_dir)

    # Add the sample to the dataset and save
    if dataset:
        logging.info("Adding sample to dataset")
        dataset.add_sample(sample)
        sample.save()
        logging.info(f"Adding sample finished...")

        logging.info(f"Exporting dataset into {dataset_path}")
        dataset.export( export_dir=dataset_path, dataset_type=dataset_type)
        logging.info(f"Exporting dataset finished...")

    progress_bar.close()
    out.release()
    cap.release()

    
def run_compress_video(input_path, output_path, size_upper_bound = 0, bitrate = 0):
    """
    Background task to run the compress_video function.
    """
    try:
        # Run the video processing function
        path, extension = os.path.splitext(output_path)
        extension = '.mp4'
        output_mp4_path = path + extension
        if size_upper_bound:
            output_file_path = compress_video_to_size(input_path, output_mp4_path, size_upper_bound)
        elif bitrate:
            output_file_path = compress_video_to_bitrate(input_path, output_mp4_path, bitrate, 0)
        else:
            logging.error("🚨 Error: run_compress_video - no size or bitrate provided")

        logging.info("Video compression completed - {output_file_path}.")

    except Exception as e:
        logging.error(f"🚨 Error during video compression: {e}")
        traceback.print_exc()


def run_process_video(  model_path, input_path, output_path, dataset_path=None, 
                        tracker=None, embedder=None, embedder_wts=None,
                        tile=None, 
                        start_ms=None, end_ms=None, 
                        conf=None,
                        cvat=False
                    ):
    """
    Background task to run the process_video function.
    """
    if not tracker: tracker="botsort.yaml"

    try:
        if os.path.isdir(output_path):
            start = round( start_ms / 1_000, 2) if start_ms else None
            end   = round( end_ms   / 1_000, 2) if end_ms   else None
            embedder_name = embedder.replace('/','-') if embedder else None

            output_name = build_name([input_path, model_path, tracker, tile, embedder_name, start, end] )
            output_path = os.path.normpath(output_path) + '/' + output_name + '.mp4'

        path, extension = os.path.splitext(output_path)
        
        target = ".avi" if extension == ".avi" else ".mp4"
        output_path = path + ".avi"
       
        logging.info(f"target {target} output_path {output_path}")
        logging.info(f"tracker {tracker} tile {tile}")
        logging.info(f"embedder {embedder} embedder_wts {embedder_wts}")

        # Run the video processing function
        process_video(  model_path, process_one_frame, 
                        input_path, output_path, dataset_path,
                        tracker, embedder, embedder_wts,
                        tile, 
                        start_ms, end_ms,
                        conf,
                        cvat)

        if target == ".mp4": 
            video_bitrate, audio_bitrate = get_bitrate(input_path)
        
            if  video_bitrate < 2 * 1024 * 1024:
                video_bitrate = 2 * 1024 * 1024

            logging.info(f"bitrate video:{video_bitrate} audio:{audio_bitrate}")

            path, extension = os.path.splitext(output_path)
            extension = '.mp4'
            output_mp4_path = path + extension
            output_mp4_path = compress_video_to_bitrate(output_path, output_mp4_path, video_bitrate, audio_bitrate)

            if (output_mp4_path):
                logging.info(f"removing {output_path}")
                os.remove(output_path)
                logging.info(f"keeping  {output_mp4_path}")

        logging.info("Video processing completed.")
    except Exception as e:
        logging.error(f"🚨 Error during video processing: {e}")
        traceback.print_exc()


usage = '''Usage : http://localhost:8080/process?config_name=yolo11_sliced&input_path=./input/videoplayback.mp4&output_path=./output&start_ms=180000&end_ms=182000
        Supported configurations : yolo11_sliced | yolo11
        Mapped voiumes: input, output and models
        To (re)start : docker-reset.sh
    '''

@app.get("/")
async def root():
    return { "message": usage }


@app.get("/bitrate")
async def bitrate(input_path: str):
    """
    Endpoint to return video bitrate using a GET request.
    """
    video_bitrate, audio_bitrate, streams = get_bitrate(input_path)
    return { "video_bitrate": video_bitrate, "audio_bitrate" : audio_bitrate, "streams" : streams }


@app.get("/compress")
async def compress_video(
    background_tasks: BackgroundTasks,
    input_path: str,
    output_path: str,
    size: Optional[int] = 0,
    bitrate: Optional[int] = 0,
):
    """
    Endpoint to start video compression as a background task using a GET request.
    """
    if not size and not bitrate:
        return  {"error": "please specify not-to-exceed-size or bitrate"}

    # Validate the input path
    if not os.path.exists(input_path):
        return {"error": f"Input file not found: {input_path}"}

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
     # Add the background task
    background_tasks.add_task(
        run_compress_video,
        input_path,
        output_path,
        size,
        bitrate
    )

    return {"message": "Video processing started in the background"}


@app.get("/process")
async def process_video_in_background(
    background_tasks: BackgroundTasks,
    model_path: str,
    input_path: str,
    output_path: str,
    dataset_path: Optional[str] = None,
    tracker: Optional[str] = None,
    embedder: Optional[str] = None,
    embedder_wts: Optional[str] = None,
    tile: Optional[int] = None,
    start_ms: Optional[int] = 0,
    end_ms: Optional[int] = None,
    start: Optional[int] = 0,
    end: Optional[int] = None,
    conf: Optional[float] = None,   # minimum conf for detection 
    cvat: Optional[bool] = Query(None)  # Changed to None default
    ):
    '''
    Endpoint to start video processing as a background task using a GET request.
    '''
    use_cvat = cvat is not None

    # Validate the input path
    if not os.path.exists(input_path):
        return {"error": f"Input file not found: {input_path}"}
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # overwrite ms is sec is provided
    start_ms = start * 1_000 if start != 0         else start_ms
    end_ms   = end   * 1_000 if end   is not None  else end_ms

    # Add the background task
    background_tasks.add_task(
        run_process_video,
        model_path,
        input_path,
        output_path,
        dataset_path,
        tracker,
        embedder,
        embedder_wts,
        tile,
        start_ms,
        end_ms,
        conf,
        use_cvat
    )

    return {"message": "Video processing started in the background"}


@app.get("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_name:str,
    dataset_path: Optional[str] = None,
    testing_dataset_path: Optional[str] = None,
    model_wts: Optional[str] = None,
    num_epochs: Optional[int] = 10 ,
    freeze_features: Optional[bool] = False  # Changed to None default
):

    model_name = model_name.replace('_','-') # common typo tend to confused - with _

    if ( model_name == 'car-yaw'):
        from features.car.yaw_model import train
    else:
        msg = f'Unsupported / Unknown {model_name}'
        logging.error( msg )
        return {"message": msg }

    background_tasks.add_task(
        train,
        training_dataset_path    = dataset_path,
        testing_dataset_path    = testing_dataset_path,
        model_weights_path      = model_wts,
        freeze_features_layers  = freeze_features,
        num_epochs              = num_epochs
    )

    return {"message": f"Training of {model_name} started in the background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

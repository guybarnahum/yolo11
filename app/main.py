import cv2
import importlib
import logging
from fastapi import BackgroundTasks, FastAPI
from typing import Optional
import os
import time
from tqdm import tqdm #select best for enviroment

from compress_video import compress_video_to_size, compress_video_to_bitrate, get_bitrate, calculate_bit_rate
from yolo_classes import get_yolo_classes

app = FastAPI()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import traceback

def process_video(  detect_model, 
                    track_model, 
                    process_one_frame_func, 
                    input_path, 
                    output_path, 
                    start_ms = 0, 
                    end_ms = None):

    # Video capture
    try:
        cap = cv2.VideoCapture(input_path)
    except Exception as e:
        logger.error(f'An error occurred: {e}')
        return 

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                        cv2.CAP_PROP_FRAME_HEIGHT, 
                                        cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter( output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    # Position in video
    start_frame = int(start_ms * fps / 1000) if start_ms else 0   
    end_frame   = int(end_ms   * fps / 1000) if end_ms   else None
    frame_number= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    if not end_frame or end_frame > frame_number:
        end_frame = frame_number

    logging.debug(f"start_frame: {start_frame}, end_frame:{end_frame} ")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_ix = start_frame
    frames_to_process = end_frame - start_frame
    pct_frames = int(frames_to_process / 100) + 1

    # Loop over the video frames
    # for frame_ix in tqdm(range(start_frame,end_frame)):
    progress_bar = tqdm(total=frames_to_process)

    while not end_frame or frame_ix < end_frame :
        ret, im0 = cap.read()
        if not ret: break

        frame_ix = frame_ix + 1
        
        if ( frame_ix % pct_frames == 0 ):
            print("", end ="\n", flush=True)
            #print(progress_bar, flush=True)
        
        # logging.debug(f"frame: {frame_ix}")
         
        if end_frame and frame_ix > end_frame   : break

        im0 = process_one_frame_func( im0, detect_model, track_model)
        out.write(im0)  # write the video frame as output 

        progress_bar.update(1)
      
        # cv2.imshow("instance-segmentation-object-tracking", im0) # display
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
    progress_bar.close()
    out.release()  # release the video writer
    cap.release()  # release the video capture
    # cv2.destroyAllWindows() # destroy all opened windows
    
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
            logging.error("Error: run_compress_video - no size or bitrate provided")

        logging.info("Video compression completed - {output_file_path}.")

    except Exception as e:
        logging.error(f"Error during video compression: {e}")
        traceback.print_exc()


def run_process_video(config_name, input_path, output_path, start_ms=0, end_ms=None, image_size=1088):
    """
    Background task to run the process_video function.
    """
    try:
        if os.path.isdir(output_path):
            output_path = os. path. normpath(output_path) + "/output.mp4"

        path, extension = os.path.splitext(output_path)
        
        print(f"path: {path} extension: {extension}")

        target = ".avi" if extension == ".avi" else ".mp4"
        output_path = path + ".avi"
       
        logging.info(f"target {target} output_path {output_path}")

        # Dynamically import the specified config module
        config = importlib.import_module(config_name)
        logging.info(f'Config {config_name} loaded successfully.')

        # Load models from the config
        detect_model, track_model = config.get_models(image_size=image_size)
        process_one_frame = config.process_one_frame

        # Run the video processing function
        process_video(detect_model, track_model, process_one_frame, input_path, output_path, start_ms, end_ms)

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
        logging.error(f"Error during video processing: {e}")


usage = '''Usage : http://localhost:8080/process?config_name=yolo11_sliced&input_path=./input/videoplayback.mp4&output_path=./output&start_ms=180000&end_ms=182000&image_size=1088
        Supported configurations : yolo11_sliced | yolo11
        Mapped voiumes: input, output and models
        To (re)start : docker-reset.sh
    '''

@app.get("/")
async def root():
    return { "message": usage }


@app.get("/classes")
async def yolo_classes():
    return get_yolo_classes()


@app.get("/bitrate")
async def bitrate(input_path: str):
    """
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
        return  {"error": f"please specify not to exceed size or bitrate"}

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
    config_name: str,
    input_path: str,
    output_path: str,
    start_ms: Optional[int] = 0,
    end_ms: Optional[int] = None,
    image_size: Optional[int] = 1088
):
    """
    Endpoint to start video processing as a background task using a GET request.
    """
    # Validate the input path
    if not os.path.exists(input_path):
        return {"error": f"Input file not found: {input_path}"}
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Add the background task
    background_tasks.add_task(
        run_process_video,
        config_name,
        input_path,
        output_path,
        start_ms,
        end_ms,
        image_size
    )

    return {"message": "Video processing started in the background"}

# http://localhost:8000/process?config_name=yolo11_sliced&input_path=./input/videoplayback.mp4&output_path=./output&start_ms=180000&end_ms=182000&image_size=1088

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import modal
import logging
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse

# Create Modal app
app = modal.App("yolo-detection")

# Create models volume
volume = modal.Volume.from_name("models-vol", create_if_missing=True)
MODELS_DIR = Path("/models")

# Define base image
image = (modal.Image
         .debian_slim()
         .apt_install([
             "libgl1-mesa-glx",
             "libglib2.0-0",
             "libsm6",
             "libxext6", 
             "libxrender-dev",
             "ffmpeg"
         ])
         .pip_install_from_requirements("requirements.txt")
         .copy_local_dir("../features", "/root/features")
         .copy_local_dir("../trackers", "/root/trackers")
         .copy_local_file("../utils.py", "/root/utils.py")
         .copy_local_file("../yolo11.py", "/root/yolo11.py")
         .copy_local_file("../cvat.py", "/root/cvat.py")
         .copy_local_file("../compress_video.py", "/root/compress_video.py"))

@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    gpu="A10G",
    timeout=3600
)
def process_video(
    model_path: str,
    input_path: str, 
    output_path: str,
    dataset_path: str = None,
    tracker: str = "botsort.yaml",
    embedder: str = None,
    embedder_wts: str = None,
    tile: int = None,
    start_ms: int = 0,
    end_ms: int = None,
    conf: float = None,
    cvat: bool = False
):
    """Process video with YOLO detection and tracking"""
    import yolo11
    from utils import cuda_device, setup_model, annotate_frame
    from features.inspect import inspect
    from trackers.deepsort.tracker import setup as deepsort_setup
    from cvat import cvat_init, cvat_add_frame, cvat_save

    device = cuda_device()
    logging.info(f"Using device: {device}")

    if tracker == 'deepsort':
        deepsort_setup(embedder=embedder, embedder_wts=embedder_wts)

    # Video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")
 
    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                          cv2.CAP_PROP_FRAME_HEIGHT, 
                                          cv2.CAP_PROP_FPS))

    # Setup output writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    # Load models
    detect_model, tile_model = setup_model(model_path, tile, image_size=w, build_cls_map=True)
    
    # Frame calculations
    start_frame = int(start_ms * fps / 1000) if start_ms else 0   
    end_frame = int(end_ms * fps / 1000) if end_ms else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_to_process = end_frame - start_frame
    progress_bar = tqdm(total=frames_to_process)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if cvat:
        input_name = Path(input_path).stem
        cvat_json = cvat_init(name=input_name, video_path=input_path)

    try:
        for frame_ix in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, detections = yolo11.process_one_frame(
                frame, detect_model, tile_model, tracker, 
                tile, conf, frame_number=frame_ix, 
                device=device
            )

            # Process detections
            for detection in detections:
                if detection.inspect:
                    features = inspect(detection, frame=frame, video_path=input_path)
                    if features:
                        detections.extend(features)

            # Annotate frame
            label = f"frame_{frame_ix}"
            annotate_frame(frame, detections, label=label)

            if cvat:
                cvat_json = cvat_add_frame(frame_ix, detections, cvat_json)

            out.write(frame)
            progress_bar.update(1)

    finally:
        progress_bar.close()
        cap.release()
        out.release()

        if cvat:
            cvat_json_path = Path(output_path).with_suffix('.json')
            cvat_save(cvat_json, cvat_json_path)

@app.local_entrypoint()
def main(model_path: str, input_path: str, output_path: str, tracker: str = "botsort.yaml"):
    """Main entrypoint that accepts command line arguments"""
    process_video.remote(
        model_path=model_path,
        input_path=input_path,
        output_path=output_path,
        tracker=tracker
    )
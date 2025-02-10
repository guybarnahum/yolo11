import modal
import numpy as np
import cv2
from io import BytesIO
from modal import Image, Mount, Volume
from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from typing import Optional
import os

# Initialize Modal app and volume
app = modal.App("yolo-live-tracking")  # Changed from Stub to App
models_volume = Volume.from_name("models-vol", create_if_missing=True)  # Updated volume creation

# Create custom image with necessary dependencies (should be switched to a pre-built image)
image = (Image.debian_slim()
         .apt_install(["ffmpeg", "libsm6", "libxext6"])
         .pip_install([
             "cython",
             "python-dotenv",
             "fastapi",
             "opencv-python-headless",
             "torch",
             "torchvision",
             "torchreid",
             "deep-sort-realtime",
             "openai-clip",
             "sahi",
             "numpy"
         ]))

# Create FastAPI app
web_app = FastAPI(title="YOLO Live Tracking API")

class YOLOProcessor:
    def __init__(self):
        self.detect_model = None
        self.tile_model = None
        self.frame_count = 0
        self.initialized = False

    def initialize(self, 
                  model_path: str,
                  tracker: Optional[str] = "bytetrack.yaml",
                  embedder: Optional[str] = None,
                  embedder_wts: Optional[str] = None,
                  image_size: Optional[int] = 1088):
        """Initialize YOLO model and tracker"""
        try:
            from utils import setup_model, cuda_device
            from trackers.deepsort.tracker import setup as deepsort_setup

            # Setup DeepSORT if selected
            if tracker == 'deepsort':
                deepsort_setup(embedder=embedder, embedder_wts=embedder_wts)
            
            # Initialize model
            self.detect_model, self.tile_model = setup_model(
                model_path=model_path,
                tile=None,
                image_size=image_size,
                build_cls_map=True
            )
            
            self.tracker = tracker
            self.initialized = True
            print(f"Model initialized with {model_path} and tracker {tracker}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray, conf: Optional[float] = None) -> tuple:
        """Process a single frame using the initialized model"""
        if not self.initialized:
            raise HTTPException(status_code=400, detail="Model not initialized")

        try:
            from yolo11 import process_one_frame
            from utils import cuda_device

            self.frame_count += 1
            device = cuda_device()

            # Process frame using the existing function
            processed_frame, detections = process_one_frame(
                frame=frame,
                detect_model=self.detect_model,
                tile_model=self.tile_model,
                tracker=self.tracker,
                tile=None,
                conf=conf,
                frame_number=self.frame_count,
                device=device
            )

            return processed_frame, detections

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            raise

@app.function(  
    image=image,
    mounts=[
        Mount.from_local_dir("app", remote_path="/root/app"),
        Mount.from_local_dir("rq_worker", remote_path="/root/rq_worker"),
    ],
    volumes={"/root/models": models_volume},
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    # secrets=[
    #     modal.Secret.from_name("new-modal-secret")    
    # ]
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# Function to initialize the models volume
@app.function( 
    image=image,
    volumes={"/root/models": models_volume}
)
def init_models_volume():
    """Initialize the models volume with required model files"""
    if not os.path.exists("/root/models") or not os.listdir("/root/models"):
        print("Initializing models volume...")
        os.makedirs("/root/models", exist_ok=True)
        
        if os.path.exists("models"):
            for item in os.listdir("models"):
                src = os.path.join("models", item)
                dst = os.path.join("/root/models", item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst)
            print("Models copied to volume successfully")
        else:
            print("No local models directory found")

if __name__ == "__main__":
    # Initialize models volume when running locally
    init_models_volume.remote()
    app.serve()
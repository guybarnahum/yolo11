import modal
from pathlib import Path
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)

app = modal.App("model-setup")
volume = modal.Volume.from_name("models-vol", create_if_missing=True)

# Create image that mounts the models directory
stub_image = modal.Image.debian_slim()

EXPECTED_MODEL_FILES = [
    "yolo11n.pt",
    "yolo11l.pt", 
    "yolo11n-seg.pt",
    "license_plate_detector.pt",
    "face_detection_yunet_2023mar.onnx",
    "yolo11n-pose.pt"
]

EXPECTED_TORCH_FILES = [
    "yaw_model_weights.pth"
]

def handle_large_file_copy(src, dst):
    """Copy large binary file in chunks"""
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst, length=1024*1024)  # 1MB chunks

@app.function(
    image=stub_image,
    volumes={"/models": volume},
    mounts=[modal.Mount.from_local_dir("../../models", remote_path="/local_models")]
)
def setup_models():
    """Transfer model files from mounted directory to Modal volume"""
    
    # Create necessary directories in the volume
    os.makedirs("/models", exist_ok=True)
    os.makedirs("/models/torch", exist_ok=True)

    # Track what we've copied for validation
    copied_models = []
    copied_torch = []

    # Copy main model files
    for model_file in EXPECTED_MODEL_FILES:
        source = f"/local_models/{model_file}"
        dest = f"/models/{model_file}"
        
        if os.path.exists(source):
            try:
                src_size = os.path.getsize(source)
                logging.info(f"Copying {model_file} ({src_size/1024/1024:.1f} MB)...")
                handle_large_file_copy(source, dest)
                copied_models.append(model_file)
                
                # Verify size
                dst_size = os.path.getsize(dest)
                if dst_size != src_size:
                    raise ValueError(f"Size mismatch for {model_file}: {src_size} vs {dst_size}")
                logging.info(f"Successfully copied {model_file}")
                
            except Exception as e:
                logging.error(f"Failed to copy {model_file}: {str(e)}")
        else:
            logging.warning(f"Missing expected model file: {model_file}")

    # Copy torch weight files
    torch_dir = "/local_models/torch"
    if os.path.exists(torch_dir):
        for weight_file in EXPECTED_TORCH_FILES:
            source = os.path.join(torch_dir, weight_file)
            dest = f"/models/torch/{weight_file}"
            
            if os.path.exists(source):
                try:
                    src_size = os.path.getsize(source)
                    logging.info(f"Copying {weight_file} ({src_size/1024/1024:.1f} MB)...")
                    handle_large_file_copy(source, dest)
                    copied_torch.append(weight_file)
                    
                    # Verify size
                    dst_size = os.path.getsize(dest)
                    if dst_size != src_size:
                        raise ValueError(f"Size mismatch for {weight_file}: {src_size} vs {dst_size}")
                    logging.info(f"Successfully copied {weight_file}")
                    
                except Exception as e:
                    logging.error(f"Failed to copy {weight_file}: {str(e)}")
            else:
                logging.warning(f"Missing expected torch weight file: {weight_file}")

    logging.info("\nSetup Summary:")
    logging.info(f"Copied {len(copied_models)}/{len(EXPECTED_MODEL_FILES)} model files")
    logging.info(f"Copied {len(copied_torch)}/{len(EXPECTED_TORCH_FILES)} torch weight files")

@app.local_entrypoint()
def main():
    logging.info("Starting model setup...")
    setup_models.remote()
    logging.info("Setup complete! Run debug.py to verify the volume contents.")
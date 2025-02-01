import modal
import os

app = modal.App("debug")
volume = modal.Volume.from_name("models-vol")

@app.function(volumes={"/models": volume})
def check_volume():
    """Verify volume contents and paths"""
    import os
    from pathlib import Path
    
    print("\nChecking volume mount and contents...")
    
    # Check mount point
    if not os.path.exists("/models"):
        print("ERROR: /models directory not found!")
        return
    print("✓ /models directory exists")
    
    # Required paths
    required_paths = {
        "/models/torch/yaw_model_weights.pth": False,
        "/models/face_detection_yunet_2023mar.onnx": False,
        "/models/yolo11n-pose.pt": False,
        "/models/license_plate_detector.pt": False
    }
    
    # Check all files
    print("\nFiles in volume:")
    total_size = 0
    for root, dirs, files in os.walk("/models"):
        rel_path = root.replace("/models", "") or "/"
        print(f"\n{rel_path}")
        for f in sorted(files):
            full_path = os.path.join(root, f)
            size = os.path.getsize(full_path)
            total_size += size
            print(f"  {f:<40} {size/1024/1024:.1f} MB")
            
            # Check required files
            if full_path in required_paths:
                required_paths[full_path] = True
    
    # Validate required files
    print("\nValidating required files:")
    all_found = True
    for path, found in required_paths.items():
        status = "✓" if found else "✗"
        print(f"{status} {path}")
        if not found:
            all_found = False
    
    print(f"\nTotal size: {total_size/1024/1024:.1f} MB")
    if not all_found:
        raise Exception("Some required files are missing!")

@app.local_entrypoint()
def main():
    check_volume.remote()
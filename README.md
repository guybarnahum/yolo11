# YOLO Object Detection and Tracking System

This project uses **YOLOv11** for object detection, tracking, and processing video inputs using Docker. The system accepts video files or images as input, processes them, and outputs annotated frames with detected objects.

## Prerequisites

Ensure you have the following installed on your system:
- Docker
- Docker Compose

## Project Structure
<pre>
├── Dockerfile
├── Dockerfile_fiftyone
├── README.md
├── app
│   ├── compress_video.py
│   ├── main.py
│   ├── trackers
│   │   ├── botsort.yaml
│   │   ├── bytetrack.yaml
│   │   └── deepsort
│   │       ├── embedder_ws
│   │       │   ├── clip_ViT-B-16.pt
│   │       │   ├── clip_ViT-B-32.pt
│   │       │   └── osnet_ain_x1_0_imagenet.pth
│   │       └── tracker.py
│   ├── yolo11.py
│   └── yolo_classes.py
├── docker-compose.yml
├── docker-restart.sh
├── fiftyone_app.py
├── fiftyone_output
├── input
│   └── light-q2Lzrust_0.mp4
├── models
│   ├── yolo11l.pt
│   ├── yolo11n-seg.pt
│   ├── yolo11n-visdrone.pt
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── yolov8n.pt
├── output
│   ├── flight-q2Lzrust_0_yolo11l_deepsort_clip_ViT-B-32.mp4
│   ├── flight-q2Lzrust_0_yolo11l_deepsort_torchreid.mp4
│   ├── dataset
│   │   ├── data
│   │   │   └── flight-q2Lzrust_0.mp4
│   │   ├── labels
│   │   │   └── flight-q2Lzrust_0.json
│   │   └── manifest.json
│   ├── flight-q2Lzrust_0_yolo11l_bytetrack.mp4
│   ├── flight-q2Lzrust_0_yolo11l_deepsort.mp4
├── requirements-deepsort.txt
└── requirements.txt
 </pre>
 
## Setting Up Input and Output Directories

Before running the application, you need to create the following directories:

```bash
mkdir input output
```

input/: Place your video or image files here. These will be used as input for processing.

output/: The processed files (annotated videos or images) will be saved here.

### Building and Running the Docker Container

To build the Docker container and start the service, use the provided `docker-restart.sh` script. This script will handle the container build and restart process.

#### Running the Script

```bash
./docker-restart.sh
```

### Explanation of `docker-restart.sh`

The `docker-restart.sh` script performs the following steps:
- Stops and removes any unused containers, volumes and netwroks.
- Builds the Docker image if `resquirements.txt` was modified.
- Restarts the container using `docker-compose` with the desired service `fastapi` or `fiftyone` or both (default)

```bash
> ./docker-restart.sh fiftyone
+ echo 'Stopping containers...'
Stopping containers...
+ docker-compose down --volumes
[+] Running 3/3
 ✔ Container yolo11-tracker  Removed                                                                                                                                                 0.0s 
 ✔ Volume yolo11_mongo-data  Removed                                                                                                                                                 0.0s 
 ✔ Network yolo11_default    Removed                                                                                                                                                 0.1s 
+ echo 'Pruning unused Docker images...'
Pruning unused Docker images...
+ docker image prune -f
Total reclaimed space: 0B
+ docker network prune -f
+ echo 'Starting containers...'
Starting containers...
++ stat -f %m ./docker-restart.sh
+ last_run_time=1734175476
++ stat -f %m requirements.txt
+ last_modified_requirements=1732552420
+ build=
+ '[' 1732552420 -gt 1734175476 ']'
+ docker-compose up fiftyone
[+] Running 2/3
 ✔ Network yolo11_default       Created                                                                                                                                              0.0s 
 ✔ Volume "yolo11_mongo-data"   Created                                                                                                                                              0.0s 
 ⠙ Container yolo11-mongo-1     Created                                                                                                                                              0.1s 
 ⠋ Container yolo11-fiftyone-1  Created                                                                                                                                              0.1s 
Attaching to fiftyone-1
 100% |███████████| 4/4 [37.5s elapsed, 0s remaining, 0.1 samples/s]    
fiftyone-1  | Computing metadata...
 100% |███████| 224/224 [5.2s elapsed, 0s remaining, 49.2 samples/s]       
fiftyone-1  | App launched. Point your web browser to http://localhost:5151
fiftyone-1  | 
fiftyone-1  | Welcome to
fiftyone-1  | 
fiftyone-1  | ███████╗██╗███████╗████████╗██╗   ██╗ ██████╗ ███╗   ██╗███████╗
fiftyone-1  | ██╔════╝██║██╔════╝╚══██╔══╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔════╝
fiftyone-1  | █████╗  ██║█████╗     ██║    ╚████╔╝ ██║   ██║██╔██╗ ██║█████╗
fiftyone-1  | ██╔══╝  ██║██╔══╝     ██║     ╚██╔╝  ██║   ██║██║╚██╗██║██╔══╝
fiftyone-1  | ██║     ██║██║        ██║      ██║   ╚██████╔╝██║ ╚████║███████╗
fiftyone-1  | ╚═╝     ╚═╝╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝ v1.0.2
fiftyone-1  | 
fiftyone-1  | If you're finding FiftyOne helpful, here's how you can get involved:
fiftyone-1  | 
fiftyone-1  | |
fiftyone-1  | |  ⭐⭐⭐ Give the project a star on GitHub ⭐⭐⭐
fiftyone-1  | |  https://github.com/voxel51/fiftyone
fiftyone-1  | |
fiftyone-1  | |  🚀🚀🚀 Join the FiftyOne Slack community 🚀🚀🚀
fiftyone-1  | |  https://slack.voxel51.com
fiftyone-1  | |
fiftyone-1  | 
```

Note: make sure the script has executable permissions:

```bash
> chmod +x docker-restart.sh
```

### Accessing the Processing Endpoint

The FastAPI server runs at `http://localhost:8080`. To process a video file or an image, use the `/process` endpoint with the following query parameters.

### Endpoint Usage Example

http://localhost:8080/process?config_name=yolo11&input_path=./input/2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4&output_path=./output&image_size=1088&start_ms=180000&end_ms=240000

### Query Parameters

- **model_path** (string): Path to the model file.
- **input_path** (string): Path to the input video file.
- **output_path** (string): Path to save the processed video, if a directory is used, output file name is generated
- **dataset_path** (string, optional): path to fiftyone dataset
- **tracker** (string, optional): yolo Tracker name (e.g., ```botsort.yaml```,```bytetrack.yaml``` or ```deepsort```").
- **embedder** (string, optional) : see below for `EMBEDDER_CHOICES`
- **embedder_wts** (string, optional) : path to embedder weights file (`.pt`,`.pth`)
- **tile** (int, optional): Tile size for processing.
- **start_ms** (int, optional): Start time in milliseconds.
- **end_ms** (int, optional): End time in milliseconds.
- **start** (int, optional): Start time in seconds (overrides start_ms).
- **end** (int, optional): End time in seconds (overrides end_ms).
- **image_size** (int, optional): Image size for model processing (default: 1088).

## DeepSort Embedders

DeepSort supports the following embedder types (and also allows for externally provided embeddings)

```bash

EMBEDDER_CHOICES = [
    "mobilenet", # Default choice when embedder is None
    "torchreid",
    "clip_RN50",
    "clip_RN101",
    "clip_RN50x4",
    "clip_RN50x16",
    "clip_ViT-B/32",
    "clip_ViT-B/16",
]

#The code is searching for .pt or .pth embedder_wts in various paths, including trackers/deepsort/embedder_wts directory.

```

### CLIP models

CLIP (Contrastive Language-Image Pretraining) model well-suited for generic embedding tasks with good accuracy and moderate computational demands. May have issues diffrentiating similar objects like cars, since it is "semantic", so it marks detections as "Red small car", which may not be enough for re-id task.

- `clip_RN50` - CLIP with a ResNet-50 backbone.
- `clip_RN101` - A CLIP variant with a deeper ResNet-101 backbone.
Provides higher accuracy and is useful for scenarios requiring more detailed feature representation.
- `clip_RN50x4` - A scaled version of CLIP with a ResNet-50 backbone and 4x wider layers.
- `clip_RN50x16` - A further scaled version of CLIP with ResNet-50 and 16x wider layers.
- `clip_ViT-B/32` - A Vision Transformer (ViT) model variant with a patch size of 32.
Suitable for general-purpose ReID tasks, balance between computational cost and accuracy.
- `clip_ViT-B/16` - A Vision Transformer model variant with a smaller patch size of 16.
Offers finer-grained feature extraction and higher accuracy at the expense of increased computational requirements.

#### Installing CLIP weights

CLIP weights are not included in the repository.

OpenAI has the pt files for CLIP in the following location, rename these to `clip_XXX` (`clip_ViT-B-16.pt`) and store them in `app/trackers/deepsort/embedder_ws` directory

These are subject to change.
```bash
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
wget https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
wget https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt
wget https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

### Example Breakdown

For the example URL:
http://localhost:8080/process?model_path=./models/yolo11n.pt&input_path=./input/overlapping_walks.mp4&output_path=./output/&start=15&end=180


- **model_path**: `./models/yolo11n.pt`
- **input_path**: `input/overlapping_walks.mp4`
- **Output directory**: `output/`
- **Start time**: `15` (15 seconds)
- **End time**: `180` (3 minutes)

### Viewing the Processed Output

After processing, the output files (annotated videos or images) will be saved in the `output_path` directory as an mp4 file. You can view them using any standard media player.

The video and detections are also added to output/dataset directory where `fiftyone` app is loading it as `yolo11_dataset`


## Cleanup hints

To stop and remove all containers, networks, and volumes created by Docker Compose:

```bash
> docker-compose down --volumes
```

To clean up also images:

```bash
> docker-compose down --volumes --rmi all
```


## License

This project is licensed under the MIT License.




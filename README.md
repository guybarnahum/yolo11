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
│   ├── yolo11.py
│   └── yolo_classes.py
├── docker-compose.yml
├── docker-restart.sh
├── fiftyone_app.py
├── input
│   ├── 2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4
│   └── overlapping_walks.mp4
├── models
│   ├── yolo11l.pt
│   ├── yolo11n-seg.pt
│   ├── yolo11n-visdrone.pt
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── yolov8n.pt
├── output
│   ├── dataset
│   │   ├── data
│   │   │   ├── overlapping_walks.mp4
│   │   │   └── 2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4
│   │   ├── labels
│   │   │   ├── overlapping_walks.json
│   │   │   └── 2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.json
│   │   └── manifest.json
│   ├── overlapping_walks_yolo11n_seg.mp4
│   ├── overlapping_walks_yolo11n.mp4
│   └── overlapping_walks_yolo11s_botsort.mp4
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
- Stops and removes any running containers for this project.
- Builds the Docker image.
- Restarts the container using `docker-compose`.

Ensure the script has executable permissions:

```bash
chmod +x docker-restart.sh
```

### Accessing the Processing Endpoint

The FastAPI server runs at `http://localhost:8080`. To process a video file or an image, use the `/process` endpoint with the following query parameters.

### Endpoint Usage Example

http://localhost:8080/process?config_name=yolo11&input_path=./input/2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4&output_path=./output&image_size=1088&start_ms=180000&end_ms=240000

### Query Parameters

- **model_path** (string): Path to the model file.
- **input_path** (string): Path to the input video file.
- **output_path** (string): Path to save the processed video, if a directory is used, output file name is generated
- **tracker** (string, optional): yolo Tracker name (e.g., "botsort.yml").
- **tile** (int, optional): Tile size for processing.
- **start_ms** (int, optional): Start time in milliseconds.
- **end_ms** (int, optional): End time in milliseconds.
- **start** (int, optional): Start time in seconds (overrides start_ms).
- **end** (int, optional): End time in seconds (overrides end_ms).
- **image_size** (int, optional): Image size for model processing (default: 1088).

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

## Cleanup

To stop and remove all containers, networks, and volumes created by Docker Compose:

```bash
docker-compose down --volumes
```


## License

This project is licensed under the MIT License.




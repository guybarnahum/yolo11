# YOLO Object Detection and Tracking System

This project uses **YOLOv11** for object detection, tracking, and processing video inputs using Docker. The system accepts video files or images as input, processes them, and outputs annotated frames with detected objects.

## Prerequisites

Ensure you have the following installed on your system:
- Docker
- Docker Compose

## Project Structure
<pre>
├── Dockerfile
├── app
│   ├── main.py
│   ├── yolo11.py          
│   └── yolo11_sliced.py
├── docker-compose.yml
├── docker-restart.sh
├── input
│   ├── 2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4
│   └── videoplayback.mp4
├── models
│   ├── yolo11n-seg.pt
│   ├── yolo11n.pt
│   └── yolov8n.pt
├── output
│   ├── output_120_180.avi
│   ├── output_15_60.avi
│   ├── output_180_240.avi
│   └── output_60_120.avi
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

- **config_name**: The configuration to use (e.g., `yolo11`). This tells the system which model configuration to load.
- **input_path**: The relative path to the input file (video or image) located in the `input/` directory.
- **output_path**: The relative path to the directory where processed files should be saved. Typically set to `./output`.
- **image_size**: The resolution to which the input frames should be resized (e.g., 1088).
- **start_ms** (optional): The starting timestamp (in milliseconds) from where processing should begin.
- **end_ms** (optional): The ending timestamp (in milliseconds) where processing should stop.

### Example Breakdown

For the example URL:

http://localhost:8080/process?config_name=yolo11&input_path=./input/2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4&output_path=./output&image_size=1088&start_ms=180000&end_ms=240000

- **Model**: `yolo11`
- **Input file**: `input/2024_10_29_TD2eeqFV_b246bddda14e6e760189eea14480b3f8_flight-TD2eeqFV_0.mp4`
- **Output directory**: `output/`
- **Image size**: `1088`
- **Start time**: `180000ms` (3 minutes)
- **End time**: `240000ms` (4 minutes)

### Viewing the Processed Output

After processing, the output files (annotated videos or images) will be saved in the `output/` directory. You can view them using any standard media player.

## Cleanup

To stop and remove all containers, networks, and volumes created by Docker Compose:

```bash
docker-compose down --volumes
```


## License

This project is licensed under the MIT License.




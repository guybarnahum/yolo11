FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libcurl4 wget && \
    rm -rf /var/lib/apt/lists/*


# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a directory for pip cache (optional, pip will create it if it doesn't exist)
ENV PIP_CACHE_DIR=/pip_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt 
# RUN pip install --no-cache-dir -r requirements.txt 

RUN pip install --cache-dir=$PIP_CACHE_DIR -U ultralytics

COPY requirements-yolo.txt .
RUN pip install --cache-dir=$PIP_CACHE_DIR -r requirements-yolo.txt 

RUN pip install --upgrade --force-reinstall mediapipe
RUN wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

COPY . .

EXPOSE 8080

# Run the FastAPI application using uvicorn
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

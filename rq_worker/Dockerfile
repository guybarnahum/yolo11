FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libcurl4 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a directory for pip cache (optional, pip will create it if it doesn't exist)
ENV PIP_CACHE_DIR=/pip_cache

WORKDIR /rq_worker

RUN pip install --cache-dir=$PIP_CACHE_DIR -U redis
RUN pip install --cache-dir=$PIP_CACHE_DIR -U rq

COPY rq_worker.py .

# Run the worker
CMD ["python", "rq_worker.py"]

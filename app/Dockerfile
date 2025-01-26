FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libcurl4 wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    python3-dev \
    autoconf \
    automake \
    libtool \
    libffi-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    export PATH=$PATH:/root/.cargo/bin && \
    rustup default stable

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a directory for pip cache (optional, pip will create it if it doesn't exist)
ENV PIP_CACHE_DIR=/pip_cache

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt 
# RUN pip install --no-cache-dir -r requirements.txt 

RUN pip install --cache-dir=$PIP_CACHE_DIR -U ultralytics

COPY requirements-yolo.txt .
# Install PaddlePaddle, Tensorboard first due to conflicts with mmcv
RUN pip install --cache-dir=$PIP_CACHE_DIR paddlepaddle
RUN pip install --cache-dir=$PIP_CACHE_DIR tensorboard
RUN pip install --cache-dir=$PIP_CACHE_DIR -r requirements-yolo.txt 

COPY . .

EXPOSE 8080

# Run the FastAPI application using uvicorn
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

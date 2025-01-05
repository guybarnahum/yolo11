FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libcurl4 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV SAM2_BUILD_ALLOW_ERRORS=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install -U ultralytics

# Install SAM2
RUN python3 -m pip install --upgrade pip && \
    git clone https://github.com/facebookresearch/sam2.git /tmp/sam2 && \
    cd /tmp/sam2 && \
    python3 -m pip install -e . && \
    python3 -c "import sam2; print('SAM2 installed successfully')" && \
    cd /app

# Copy application code
COPY app/ .

EXPOSE 8080

CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
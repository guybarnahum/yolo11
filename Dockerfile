FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*


# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install -U ultralytics

COPY app/ .

EXPOSE 8080

# Run the FastAPI application using uvicorn
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

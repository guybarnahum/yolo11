FROM voxel51/fiftyone:latest

WORKDIR /app
COPY fiftyone_app.py .

# Install missing dependencies
RUN apt-get update && apt-get install -y libcurl4

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install watchdog

# RUN fiftyone migrate --all -v 1.0.1
# Run the FastAPI application using uvicorn
CMD ["python", "fiftyone_app.py"]

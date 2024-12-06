import fiftyone as fo
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# The directory containing the data to import
dataset_name = "yolo11_dataset"
dataset_dir = "/app/output/dataset"

# The type of data being imported
dataset_type = fo.types.FiftyOneVideoLabelsDataset

# Initialize the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=dataset_name,
    overwrite=True
)

# Print available fields in the dataset
logging.warning(f"Loaded dataset: {dataset}")
logging.warning(dataset.get_field_schema())
logging.warning(dataset.get_frame_field_schema()) 

# Create a custom App config
app_config = fo.app_config.copy()
app_config.color_by = "instance"

# Launch FiftyOne session
global session
session = fo.launch_app(dataset, config=app_config)

# File system event handler
class DatasetChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global session
        if event.is_directory:
            return  # Only interested in file changes
        logging.info(f"Detected change: {event.src_path}")
        
        # Refresh session to show updates
        session.refresh()

# Watch directory for changes
def watch_dataset_dir(path):
    event_handler = DatasetChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

# Start watching dataset directory
watch_dataset_dir(dataset_dir)

session.wait()  # Keep the session open
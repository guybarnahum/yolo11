import fiftyone as fo
from fiftyone import ViewField as F

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

# Create a custom App config
app_config = fo.app_config.copy()
app_config.color_by = "instance"

# Launch FiftyOne session
global session

# Set up clip view
clips = dataset.to_clips("frames.detections")
clips.compute_metadata()

# Notice: compute metadata above sets metadata.total_frame_count to the original video frame count, 
# and not the clip frame count(!)
# Need to fix it so we can filter based on metadata.total_frame_count
for index, clip in enumerate(clips):
    clip_frames = len(clip.frames)
    clip.metadata.total_frame_count = len(clip.frames)
    clip.save()  # Save changes to the clip

# Filter clips with at least 10 frames
view = clips.match(F("metadata.total_frame_count") >= 60)
session = fo.launch_app(view, config=app_config)
# session.view = clips

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
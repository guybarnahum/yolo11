import logging

from .car.inspect    import inspect as inspect_car
from .person.inspect import inspect as inspect_person

def inspect(detection, frame, video_path):
    
    features = []
    if detection.name == 'car':
        features = inspect_car( detection, frame, video_path)
    elif detection.name == 'person':
        features = inspect_person(detection, frame, video_path)
    else:
        logging.error(f"can't inspect detection type :{detection.name}")

    logging.debug(f"inspection_job ended for {detection} found {features}")
    return features

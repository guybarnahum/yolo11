import logging

def detect_pose_from_bbox(bbox):
    
    # Bounding box dimensions
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
 
    # Rough yaw estimation based on aspect ratio
    aspect_ratio = width / height
    
    # Typical car aspect ratios by orientation
    if aspect_ratio < 1.2:  # Nearly square (front/back view)
        yaw = 0  # or 180 degrees
    elif aspect_ratio > 2:  # Wide (side view)
        yaw = 90 # or 270 degrees
    else :
        yaw = 45 
    return yaw


from .yaw_model  import predict_car_yaw

def pose_car_yaw(car_detection, car_frame):
    
    yaw = predict_car_yaw(car_frame)
    return yaw




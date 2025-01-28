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


from .yaw_model  import setup_yaw_model, predict_yaw

car_yaw_model = setup_yaw_model('./models/torch/yaw_model_weights.pth')

def pose_car_yaw(car_detection, car_frame):
    global car_yaw_model
    yaw = predict_yaw(car_yaw_model, car_frame)
    return yaw




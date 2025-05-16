import logging

from .yaw_model_onnx  import predict_car_yaw

def pose_car_yaw(car_detection, car_frame):
    
    yaw = predict_car_yaw(car_frame)
    return yaw




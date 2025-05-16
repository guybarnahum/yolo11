import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

import onnxruntime  as ort

class EfficientNetONNX:
    def __init__(self, model_path):
        
        self.input_shape = (224, 224)  # EfficientNet-B0 default input size
        self.class_names = ['0','45','90','135','180','225','270','315']

        try:
            self.providers  = self.get_available_providers()
            self.session    = ort.InferenceSession(model_path, providers=self.providers)    
        except Exception as e:
            logging.error(f'EfficientNetONNX init error: {str(e)}')
            self.session = False

        # Correct Transform Order
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert from NumPy (cv2 frame) to PIL
            transforms.Resize(self.input_shape),  # Resize while still a PIL Image
            transforms.ToTensor(),  # Convert to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_available_providers(self):
        '''
        Detects whether GPU (CUDA) is available and returns the best execution provider.
        '''
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            logging.info('EfficientNetONNX Using GPU (CUDA) for inference')
            return ['CUDAExecutionProvider']
        else:
            logging.info('EfficientNetONNX Using CPU for inference')
            return ['CPUExecutionProvider']

    def preprocess(self, frame):
        """Preprocess OpenCV frame for ONNX inference"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        tensor = self.transform(frame).unsqueeze(0)  # Add batch dimension
        return tensor.numpy()

    def predict(self, frame):
        if not self.session: 
            logging.warning(f'EfficientNetONNX - no onnxrealtime session')
            return False

        """Run inference on an OpenCV frame"""
        input_tensor = self.preprocess(frame)
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        output = self.session.run(None, ort_inputs)
        
        # Apply softmax to convert logits to probabilities
        logits = output[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
    
        # Get the class with highest probability
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        class_name = self.class_names[predicted_class]

        logging.debug(f'logits : {logits}')
        logging.debug(f'probabilities : {probabilities}')
        logging.debug(f'class_name : {class_name}')
        logging.debug(f'confidence : {confidence}')

        return class_name, confidence


car_yaw_wts_path = "./models/torch/yaw_model_weights.onnx"
car_yaw_model    = EfficientNetONNX( car_yaw_wts_path ) 

def predict_car_yaw(car_frame):

    class_name, conf = car_yaw_model.predict(car_frame)
    return class_name, conf

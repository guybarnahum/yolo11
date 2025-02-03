import cv2
import logging
from types import MethodType
from utils import setup_model, cuda_device, flatten_results, print_detections, annotate_frame, annotate_frame_text
from .pose import detect_pose_from_bbox, pose_car_yaw

license_plate_model,_ = setup_model('./models/license_plate_detector.pt')

def ocr_paddle_read(self, lp_frame):
    results = self.ocr(lp_frame, cls=True)
    logging.debug(f'ocr_paddle_read: results {results}')

    lps = []
    if results and results[0]:  # Check if results and results[0] are not None
        try:
            for result in results[0]:
                if result:
                    text, score = result[1]
                    lps.append([text, score])

        except Exception as e:
            logging.error(f"Error ocr_paddle_read - {str(e)}")
            logging.error(f"results : {results}")
            lps = []

    logging.debug(f'ocr_paddle_read: lps {lps}')
    return lps

def ocr_easy_read(self, lp_frame):
    
    results = ocr_easy.readtext(lp_frame, decoder='beamsearch')
    '''
    Sort from results from left to right - lp[0] being the bbox, lp[0][0] is the top-left, 
    lp[0][0][0] being the left of the top-left of the bbox
    
    [
        [
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # Bounding box (4 points)
            'Detected Text',                           # Text content
            Confidence                                 # Confidence score (float)
        ],
        ...
    ]
    '''
    results.sort(key=lambda lp: lp[0][0][0]) 
    
    lps = []
    for result in results:
        _, text, score = result
        lp.append([text,score])

    return lps

def setup_ocr(ocr_select = None):

    if not ocr_select: return None

    if ocr_select == 'easyocr':
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        ocr_reader.perform_ocr = MethodType(ocr_easy_read,ocr_reader)
    
    elif ocr_select == 'paddleocr':
        from paddleocr import PaddleOCR
        logging.getLogger("ppocr").setLevel(logging.INFO)

        ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
        ocr_reader.perform_ocr = MethodType(ocr_paddle_read,ocr_reader)
        
    else:
        logging.error(f'Unsupported ocr_select: {ocr_select}')
        ocr_reader = None

    # Save ocr tyoe in ocr_reader object
    if  ocr_reader:
        ocr_reader.type = ocr_select

    return ocr_reader

ocr_reader = setup_ocr(ocr_select='paddleocr')

def ocr_license_plate( lp_frame, min_score = 0.7 ):
    '''
    OCR the license plate text from the given image.

    Args:
        lp_frame (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the license plate text and confidence score.
    '''
    global ocr_reader
    lps = ocr_reader.perform_ocr(lp_frame)

    # Print results
    # for ix,lp in enumerate(lps):
    #    print(f"{ix}> lp: {lp}")
        
    if not len(lps):
        logging.debug(f'ocr_license_plate: no text found')
        return None, None

    lps_text, lps_score = '', 0.0
    
    if len(lps) == 1:
        lps_text, lps_score = lps[0]
    else:
        for ix, lp in enumerate(lps):
            # print(f'{ix}> ip: {lp}')
            text, score = lp
            lps_text = lps_text + ' ' + text
            # total_score is sqrt(sum(score^2))
            lps_score = lps_score + score * score
        
        lps_score = lps_score ** 0.5

    if lps_score < min_score:
        lps_text, lps_score = None, None

    logging.debug(f'ocr_license_plate: {lps_text},{lps_score}')
    return lps_text, lps_score


def read_license_plate( lp_detection, car_frame, min_score = 0.7 ):

    x1, y1, x2, y2 = lp_detection.bbox
    lp_frame = car_frame[ int(y1): int(y2), int(x1): int(x2), :]
    
    logging.debug(f"read_license_plate -({x1},{y1}) x ({x2},{y2}) - lp_frame :{lp_frame}")

    lp_frame = cv2.cvtColor(lp_frame, cv2.COLOR_BGR2GRAY)
    lp_frame = cv2.bilateralFilter(lp_frame, 11, 17, 17)
    lp_frame = cv2.adaptiveThreshold(lp_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    # lp_frame = cv2.cvtColor(lp_frame, cv2.COLOR_BGR2GRAY)
    # _, lp_frame = cv2.threshold(lp_frame, 64, 255, cv2.THRESH_BINARY_INV)

    return ocr_license_plate(lp_frame,min_score=min_score)
    

def should_inspect(detection):
    '''
    Criteria for inspection 
    '''
    return detection.area > 400 * 400


def inspect(car_detection, frame, video_path):
    '''
        Returns an array of detections
    '''
    global license_plate_model
    global ocr_reader

    device = cuda_device()
    if car_detection.name != 'car':
        logging.warning( f'features.car.inspect called for {car_detection}')
        return []

    x1, y1, x2, y2 = car_detection.bbox
    car_frame = frame[ int(y1): int(y2), int(x1): int(x2), :]

    yaw, conf = pose_car_yaw(car_detection, car_frame)
    yaw_label = f'yaw:{yaw},{conf:.3f}'
    
    car_detection.detail = yaw_label

    car_detection.attributes.append({"name":"yaw", "value":yaw})
    car_detection.attributes.append({"name": "yaw_conf", "value":conf})

    logging.debug(f'features.car.inspect {yaw_label}')

    # License Plate(s)
    results = license_plate_model.predict( source=car_frame, verbose=False, device=device )
    lp_detections = []

    if len(results[0].boxes):

        # Offset detection back to original frame from car frame
        lp_detections = flatten_results(results, frame_number=car_detection.frame_number, offset_x=x1, offset_y=y1)
    
        
        for lp in lp_detections:
            
            lp.name     = "license_plate"
            lp.track_id = car_detection.track_id

            if ocr_reader:
                
                lp_text, lp_score = read_license_plate( lp, frame ) # lp bbox is in frame ccordinates

                if lp_text and lp_score and lp_score > 0.5:

                    lp_score  = round(lp_score,2)
                    lp.detail = f"{lp_text}({lp_score:.2f})"

                    logging.debug( f'features.car.inspect detail : {lp.detail}' )

                    lp.attributes.append({"name": "ocr_text","value":lp_text})
                    lp.attributes.append({"name": "ocr_text_conf","value":lp_score})

    return lp_detections

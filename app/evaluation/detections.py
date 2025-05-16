
import logging
from pprint import pprint, pformat
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import defaultdict
import Levenshtein
from .groundtruth import gt_get_frame_detections
from utils import print_detection, print_detections

class DetectionQualityEvaluator:
    def __init__(self, gt = None):
        self.frames = defaultdict(dict)
        self.gt = gt

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


    def _prepare_precision_data(self, matched_pairs, unmatched_preds):
        '''
        To compute the precision-recall curve we need to construct:

            y_true  : Binary labels (1 for correct matches, 0 for unmatched predictions).
            y_scores: Confidence scores, which can be the IoU values from matched pairs and assign 0 (or very low value) 
                    to unmatched predictions.
                    
        From our matches:
        
        - For Matched Pairs:
            y_true = 1 (since these are valid matches).
            y_score = IoU (calculated during matching).

        - For Unmatched Predictions:
            y_true = 0 (false positives).
            y_score = 0 (since there's no overlap).

        '''
        y_true = []
        y_scores = []

        # For matched pairs
        for pred_idx, gt_idx, iou, _, _ in matched_pairs:
            y_true.append(1)         # Correct match
            y_scores.append(iou)     # IoU as the confidence score

        # For unmatched predictions (false positives)
        for pred_idx in unmatched_preds:
            y_true.append(0)         # No match = false positive
            y_scores.append(0)       # No IoU, assign zero

        if y_true == []:
            logging.debug(f'y_true is empty - matched_pairs: {matched_pairs},unmatched_preds: {unmatched_preds}')

        return y_true, y_scores


    def _match_detections(self, detections, ground_truths, iou_threshold=0.5, labels=None):
        
        matched_pairs = []
        unmatched_preds  = set(range(len(detections)))
        unmatched_truths = set(range(len(ground_truths)))
        preds_num = 0

        for pred_idx, detection in enumerate(detections):
            
            label = detection.name

            # filter labels
            if labels and label not in labels:
                logging.debug(f'skipping {label} not in {labels}')
                unmatched_preds.discard(pred_idx)
                continue

            preds_num += 1
            bbox  = detection.bbox

            # Look for a groud_truth match with bbox and label
            best_iou = 0
            best_gt_idx = None

            # look for a match:
            for gt_idx, ground_truth in enumerate( ground_truths ):
                
                gt_label = ground_truth.name
                if gt_label != label:
                    logging.debug(f'looking for {label} - skipping {gt_label}')
                    unmatched_truths.discard(gt_idx)
                    continue

                gt_bbox = ground_truth.bbox
                iou = self.calculate_iou(bbox, gt_bbox)

                if iou > best_iou and iou >= iou_threshold:
                    best_iou    = iou
                    best_gt_idx = gt_idx

            # we have a match!
            if best_gt_idx is not None: 
                    
                pred_id = detections   [pred_idx   ].track_id
                true_id = ground_truths[best_gt_idx].track_id

                matched_pairs.append((pred_idx, best_gt_idx, best_iou, pred_id, true_id ))
                unmatched_preds.discard(pred_idx)
                unmatched_truths.discard(best_gt_idx)
        
        # <--- need to debug this
        for idx in unmatched_preds:
            logging.warning(f'🐞 unmatched_preds: idx:{idx}')
            print_detection( detections[idx] )
            
        return matched_pairs, unmatched_preds, unmatched_truths, preds_num


    def _calculate_detection_rate(self, matched_pairs, unmatched_truths):
        '''
            The Detection Rate (a.k.a Recall) in object detection/tracking tasks. 
            It measures the proportion of ground truth objects that were correctly detected.

            Detection Rate = True Positives(TP) / True Positives (TP) + False Negatives (FN)
 
            Where:

            True Positives  (TP): Correctly matched predictions (i.e., our matched pairs).
            False Negatives (FN): Ground truth objects that were missed (i.e., unmatched ground truths).

        '''

        true_positives  = len(matched_pairs)
        false_negatives = len(unmatched_truths) # Unmatched ground truths

        detection_rate = 0
        if (true_positives + false_negatives) > 0:
            detection_rate = true_positives / (true_positives + false_negatives) 
    
        return round(detection_rate,3)

    def evaluate_no_matches( self, pred_len ):

        '''
            pred exist -> Predictions exist without ground truth (false positives) 
            pred do not exist -> No predictions and no ground truth (perfect case), 
                There were no objects to detect, and the model didn’t produce any false positives.
        '''
        
        val = 0.0 if pred_len > 0 else 1.0
        
        '''
            When we have False Positives (FP), mota is 0.0:
            
                mota = 1 - (FP + FN + ID_Switches) / max(1, groundtruth_len)

            or in our case : 
                
                mota = 1 - (FP/1) or 1 - FP 

            Sicne FP is at least 1, we set mota to 0.0 since negative mota is meaningless
        '''
        return {
            'precision'         : [val],
            'recall'            : [val],
            'average_precision' :  val,
            'mean_iou'          :  val,
            'detection_rate'    :  val,

            'tracking'          : { 'false_negatives' : 0,
                                    'false_positives' : pred_len,
                                    'groundtruth_len' : 0,
                                    'id_switches'     : 0,
                                    'fragmentations'  : pred_len, # false_negatives + false_positives
                                    'mota'            : val
                                }
        }

    def evaluate_object_detection(self, detections, ground_truths, iou_threshold=0.5, labels=None):
        '''
            Evaluate YOLO object detection performance
        '''

        # matched_pairs are an array of (pred_ix, gt_ix, iuo, detection_track_id, truth_track_id) topples
        matched_pairs, unmatched_preds, unmatched_truths, pred_len = self._match_detections( detections, ground_truths, iou_threshold=iou_threshold, labels=labels)
        y_true, y_scores = self._prepare_precision_data(matched_pairs, unmatched_preds)
        
        if not len(y_true): # we have no matches!
            if pred_len > 0:
                logging.warning(f'evaluate_object_detection - False Positives for {labels}') 
                print_detections(detections,pre=f'detections type {labels}',labels=labels)
                print_detections(ground_truths,pre=f'ground_truths type {labels}',labels=labels)
            else:
                logging.debug(f'evaluate_object_detection - No predictions and no ground truth (perfect case) for {labels}')

            return self.evaluate_no_matches(pred_len)

        # We have matches!
        # Precision stats when ground truth exists
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        # iou mean value
        matched_iou_array = [ t[2] for t in matched_pairs]
        logging.debug(f'matched_iou_array : {matched_iou_array}')

        if np.any(np.isnan(matched_iou_array)):
            logging.warning(f'evaluate_object_detection - nan value in iou matched_pairs {matched_pairs}')

        mean_iou = np.nanmean(matched_iou_array)
        if not mean_iou or np.isnan( mean_iou): mean_iou = 0
        mean_iou = round(mean_iou, 3)

        # Detection Rate A.k.a Recall - see _calculate_detection_rate above 
        detection_rate = self._calculate_detection_rate(matched_pairs, unmatched_truths)

        # Tracking metrics
        tracking = self.evaluate_tracking(matched_pairs, unmatched_preds, unmatched_truths)

        return {
            'precision'         : precision,
            'recall'            : recall,
            'average_precision' : average_precision,
            'mean_iou'          : mean_iou,
            'detection_rate'    : detection_rate,
            'tracking'          : tracking
        }


    def evaluate_face(self, detections, ground_truths, iou_threshold=0.5):
        
        '''
        Evaluate YuNet face detection performance
        '''

        return self.evaluate_object_detection(  detections, 
                                                ground_truths, 
                                                iou_threshold=iou_threshold, 
                                                labels={'face'}
                                            )
        
        
    def evaluate_license_plate(self, detections, ground_truths, iou_threshold=0.5):
        '''
            Evaluate license plate detection 
        '''
        return self.evaluate_object_detection(  detections, 
                                                ground_truths, 
                                                iou_threshold=iou_threshold, 
                                                labels={'license_plate'}
                                            )
        
        '''
        def evaluate_ocr(self, ocr_texts, true_texts):
            """Evaluate PaddleOCR performance"""
            char_accuracies = []
            word_accuracies = []
            
            for ocr_text, true_text in zip(ocr_texts, true_texts):
                # Character-level accuracy
                char_dist = Levenshtein.distance(ocr_text, true_text)
                char_acc = 1 - char_dist / max(len(true_text), 1)
                char_accuracies.append(char_acc)
                
                # Word-level accuracy
                ocr_words  = ocr_text.split()
                true_words = true_text.split()
                correct_words = sum(1 for p, t in zip(ocr_words, true_words) if p == t)
                word_acc = correct_words / max(len(true_words), 1)
                word_accuracies.append(word_acc)
            
            self.metrics['ocr'].update({
                'char_accuracy': np.mean(char_accuracies),
                'word_accuracy': np.mean(word_accuracies)
                })
        '''


    def _calculate_mota(self, false_negatives, false_positives, id_switches, ground_truths_count):
        '''
        Calculate Multiple Object Tracking Accuracy (MOTA).
        
        Parameters:
            false_negatives ( len(unmatched_truths) ) )
            false_positives ( len(unmatched_preds   ) )
            id_switches (int): Number of ID switches.
            ground_truths_count (int): Total number of ground truth objects.
        
        Returns:
            float: MOTA score.
        '''

        logging.debug(f'_calculate_mota - false_negatives:{false_negatives}, false_positives:{false_positives}, id_switches: {id_switches}, gt_count:{ground_truths_count}')

        mota = 1 - (false_negatives + false_positives + id_switches) / max(1, ground_truths_count)
        return round(mota,3)


    def evaluate_tracking(self,  matched_pairs, unmatched_preds, unmatched_truths):
        """Evaluate tracking performance without assuming matching track IDs."""
        
        gt_len      = len(matched_pairs) + len(unmatched_truths)
        id_switches = 0
        fragmentations = 0
        prev_gt_id_map = {}

        for pred_idx, gt_idx, _, pred_id, true_id in matched_pairs:
            
            if true_id in prev_gt_id_map and prev_gt_id_map[true_id] != pred_id:
                id_switches += 1
            prev_gt_id_map[true_id] = pred_id

        
        false_negatives = len(unmatched_truths)  
        false_positives = len(unmatched_preds )   

        fragmentations  = false_positives + false_negatives
        
        mota = self._calculate_mota(false_negatives, 
                                    false_positives,
                                    id_switches    , 
                                    ground_truths_count = gt_len
                                )
        return {
            'false_negatives' : false_negatives,
            'false_positives' : false_positives,
            'groundtruth_len' : gt_len,
            'id_switches'     : id_switches,
            'fragmentations'  : fragmentations,
            'mota'            : mota 
        }
    

    def evaluate_one_frame(self, detections, ground_truths, frame_number, store = True):
        
        frame_metrics = {}
        res = self.evaluate_object_detection(detections,ground_truths, labels={'car','person','bicycle'})
        frame_metrics['yolo'] = res

        res = self.evaluate_face(detections,ground_truths)
        frame_metrics['yunet'] = res

        res = self.evaluate_license_plate(detections,ground_truths)
        frame_metrics['license_plate'] = res

        if store:
            self.frames[frame_number] = frame_metrics

        return frame_metrics


    def get_frame_metrics(self, frame_number):
        frame_metrics = None

        if  frame_number in self.frames:
            frame_metrics = self.frames[frame_number]
        else:
            logging.warning(f'get_frame_metrics - no metrics for {frame_number}')
        
        return frame_metrics


'''
    Evaluate interface : 
        
        init with groudtruth object
        evaluate_frame
        evaluate_aggregate_metrics
        terminate
'''

def evaluate_init(gt):
    ev = DetectionQualityEvaluator()
    ev.gt = gt
    return ev


def evaluate_frame( ev, detections, frame_number ):

    ground_truths = gt_get_frame_detections(ev.gt,frame_number)
    logging.debug(f'frame: {frame_number} - groudtruth detections: {ground_truths}')
    metrics = ev.evaluate_one_frame(detections,ground_truths,frame_number)
    return metrics
   

def evaluate_aggregate_metrics(ev, frames=None):
    
    # Make pretty prints possible in np
    np.set_printoptions(precision=3, suppress=True, formatter={'all': lambda x: f'{x:.3f}'})

    if not frames:
        frames = ev.frames.keys() # default is all the frames we have

    # keys to literate over in dict
    model_names   = ['yolo', 'yunet', 'license_plate']
    metric_fields = [ 'average_precision', 'detection_rate', 'mean_iou' ]
    tracking_metrics_fields = [ 'false_negatives','false_positives','fragmentations','groundtruth_len','id_switches']

    # 0 pass - build result object for our models
    frame_set_metrics = {}

    for model in model_names:
        frame_set_metrics[model] = {
            
            # 1st pass: collect from all frames 
            # 2nd pass: perform mean on the series
            'average_precision' : [],
            'detection_rate'    : [],
            'mean_iou'          : [],

            # 1st pass: Sum each over the frames 
            # 2nd pass: use the sums to recalculate mota
            'tracking': {'false_negatives'  : 0,
                         'false_positives'  : 0,
                         'fragmentations'   : 0,
                         'groundtruth_len'  : 0,
                         'id_switches'      : 0,

                         'mota'             : 0.0
            }
        }

    # 1st pass - collect frame metrics
    for frame_number in frames:    
        metrics = ev.get_frame_metrics(frame_number)
        if metrics:
            logging.debug(f'frame {frame_number} :\n{pformat(metrics)}')
        else:
            logging.warning(f'frame {frame_number} - not metrics found - skipped ...')
            continue

        for model in model_names:

            # 1st pass : collect all mean vals intp series
            for field in metric_fields: 
                val = metrics[model][field]
                if not np.isnan(val):
                    frame_set_metrics[model][field].append( val )
                else:
                    logging.warning(f'evaluate_aggregate_metrics - nan value at fn#{frame_number} {model}.{field}')   
            
            # 1st pass : sum tracking stats on all frames
            for field in tracking_metrics_fields: 
                val = metrics[model]['tracking'][field]
                frame_set_metrics[model]['tracking'][field] += val
    
    # 2nd pass - recalculate with aggregated frame values
    for model in model_names: 

        # 2nd pass - perform mean on series
        for field in metric_fields: 
            mean = np.nanmean(frame_set_metrics[model][field]) 
            frame_set_metrics[model][field] = round(mean, 3)

        # 2nd pass - recalculate mota with sums
        false_negatives = frame_set_metrics[model]['tracking']['false_negatives']
        false_positives = frame_set_metrics[model]['tracking']['false_positives']
        id_switches     = frame_set_metrics[model]['tracking']['id_switches']
        groundtruth_len = frame_set_metrics[model]['tracking']['groundtruth_len']

        mota = ev._calculate_mota(false_negatives, false_positives, id_switches, groundtruth_len)
        frame_set_metrics[model]['tracking']['mota'] = round(mota,3)

    # all done aggregating
    return frame_set_metrics


def evaluate_terminate(ev):
    del ev

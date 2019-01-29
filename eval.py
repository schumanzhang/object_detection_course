from tqdm import tqdm
import numpy as np
import os
import cv2


def handle_empty_indexing(arr, idx):
    if idx.size > 0:
        return arr[idx]
    return []


def compute_iou(bb_1, bb_2):

    xa0, ya0, xa1, ya1 = bb_1
    xb0, yb0, xb1, yb1 = bb_2

    intersec = (min([xa1, xb1]) - max([xa0, xb0]))*(min([ya1, yb1]) - max([ya0, yb0]))

    union = (xa1 - xa0)*(ya1 - ya0) + (xb1 - xb0)*(yb1 - yb0) - intersec

    return intersec / union


class Evaluate(object):

    def __init__(self, generator, model, config):

        self.config = config
        self.inf_model = model
        self.generator = generator

        self.class_labels = np.array(self.config["constants"]["labels"])
        self.iou_detection_threshold = self.config["model"]["iou_threshold"]

    def _find_detection(self, q_box, boxes, global_index):

        if boxes.size == 0:
            return -1

        ious = list(map(lambda x: compute_iou(q_box, x), boxes))

        max_iou_index = np.argmax( ious )

        if ious[max_iou_index] > self.iou_detection_threshold:
            return global_index[max_iou_index]

        return -1

    def _prepare_image(self, image):

        image = cv2.resize(image, (self.config["model"]["input_size"], self.config["model"]["input_size"]))
        image = image / 255.
        image = image[:,:,::-1]
        image = np.expand_dims(image, 0)

        return image

    def _make_prediction(self, image):

        output = self.inf_model.predict(image)[0]

        if output.size == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "label_idxs": np.array([])
            }

        boxes = output[:,:4]
        scores = output[:,4]
        label_idxs = output[:,5].astype(int)

        # labels = [self.class_labels[l] for l in label_idxs]

        results_dict = {
            "boxes": boxes,
            "scores": scores,
            "label_idxs": label_idxs
        }

        # print(results_dict)
        return results_dict 

    def _process_image(self, i):
        
        true_boxes, true_labels = self.generator.load_annots(i)

        image = self.generator.load_image(i)
        image = self._prepare_image(image)

        results_dict = self._make_prediction(image.copy())

        pred_boxes = results_dict["boxes"]
        conf = results_dict["scores"]
        pred_labels = results_dict["label_idxs"]

        sorted_inds = np.argsort(-conf)

        repeat_mask = [True]*len(true_boxes)
        matched_labels = []
        global_index = np.arange(len(true_labels))

        image_results = []
        image_labels = [0] * len(self.config["constants"]["labels"])

        for tl in true_labels:
            image_labels[tl] += 1

        for i in sorted_inds:

            label_mask = (pred_labels[i] == true_labels)
            index_subset = global_index[(repeat_mask)&(label_mask)]
            true_boxes_subset = true_boxes[(repeat_mask)&(label_mask)]

            idx = self._find_detection(pred_boxes[i], true_boxes_subset, index_subset)

            if idx != -1: 
                matched_labels.append(idx)
                repeat_mask[idx] = False

            image_results.append([pred_labels[i], conf[i], 1 if idx != -1 else 0])

        return image_results, image_labels

    def _interp_ap(self, precision, recall):

        if precision.size == 0 or recall.size == 0:
            return 0.

        iap = 0
        for r in np.arange(0.,1.1, 0.1):
            recall_mask = (recall >= r)
            p_max = precision[recall_mask]
            
            iap += np.max( p_max if p_max.size > 0 else [0] )

        return iap / 11

    def compute_ap(self, detections, num_gts):

        detections_sort_indx = np.argsort(-detections[:,1])
        detections = detections[detections_sort_indx]

        precision = []
        recall = []

        if num_gts == 0:
            return 0.

        for i in range(1, len(detections) + 1):

            precision.append( np.sum(detections[:i][:,2]) / i )
            recall.append( np.sum(detections[:i][:,2]) / num_gts )

        return self._interp_ap(np.array(precision), np.array(recall))

    def comp_map(self):

        detection_results = []
        detection_labels = np.array([0] * len(self.config["constants"]["labels"])) 

        for i in tqdm(range(len(self.generator.images)), desc='Batch Processed'):

            image_results, image_labels = self._process_image(i)

            detection_results.extend(image_results)
            detection_labels += np.array(image_labels)


        detection_results = np.array(detection_results)

        ap_dic = {}
        for class_ind, num_gts in enumerate(detection_labels):
            
            class_detections = detection_results[detection_results[:,0]==class_ind]            
            
            ap = self.compute_ap(class_detections, num_gts)

            ap_dic[self.class_labels[class_ind]] = ap


        return ap_dic

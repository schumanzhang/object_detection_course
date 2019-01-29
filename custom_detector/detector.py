import cv2
import os
import numpy as np
import argparse
import datetime

from utils import _decode_netout, _interval_overlap, _sigmoid, _softmax, COCO_LABELS, CUSTOM_LABELS
from keras.models import Model, load_model


class FPS:
    def __init__(self):
        # start time, end time, total number of frames
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the number of frames analysed during the start and end interval
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds
        return (self._end - self._start).total_seconds()

    def fps(self):
        # approximate frames per second
        return self._numFrames / self.elapsed()

    def get_numFrames(self):
        return self._numFrames


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)
        
    return image


def load_pretrained_model(path):
    if os.path.isfile(path):
        model = load_model(path, compile=False)

    return model


def prepare_image(image):

    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)

    return input_image

def prepare_config(model_type):

    if model_type == "coco":
        model = load_pretrained_model("saved_models/coco_model36.h5")
        LABELS = COCO_LABELS
        ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    elif model_type == "custom":
        model = load_pretrained_model("saved_models/weights.h5")
        LABELS = CUSTOM_LABELS
        ANCHORS = [1.69,1.54, 3.32,2.51, 3.41,5.53, 5.90,7.50, 6.41,4.14]

    return model, LABELS, ANCHORS

if __name__ == "__main__":

    '''
    we can use custom model or coco model
    '''

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', dest='model', type=str,
                        default="coco", help='Which model to use, custom or coco')
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(0)

    model, LABELS, ANCHORS = prepare_config(args.model)

    fps = FPS().start()

    while True:

        ret, image = cap.read()

        if ret:

            fps.update()

            input_image = prepare_image(image)

            netout = model.predict(input_image)

            boxes = _decode_netout(netout[0], 
                                obj_threshold=0.3,
                                nms_threshold=0.3,
                                anchors=ANCHORS, 
                                nb_class=len(LABELS))

            image = draw_boxes(image, boxes, labels=LABELS)

            cv2.imshow('Video', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    fps.stop()

    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
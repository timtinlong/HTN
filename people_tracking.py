import os
from urllib.request import urlretrieve
import tensorflow as tf
import cv2
from loguru import logger

from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track
import argparse
import configparser
import imageio

"""
adapated
Also look into using goturn for tracking: https://learnopencv.com/goturn-deep-learning-based-object-tracking/

"""

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-m', '--model', type=str, default='MobileNet')
parser.add_argument('-i', '--input_video', type=str, default='/Users/Timot/Desktop/Hackathon/HTN/sample_vid_1.mp4')
parser.add_argument('-c', '--confidence', type=float, default=0.5)


# run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

args = parser.parse_args()

if args.model == 'MobileNet':
    WEIGHTS_PATH = 'MobileNet-SSD-master/mobilenet_iter_73000.caffemodel'
    CONFIG_PATH = 'MobileNet-SSD-master/voc/MobileNetSSD_deploy.prototxt'
elif args.model == 'FaceDetect':
    WEIGHTS_PATH = '/Users/Timot/Desktop/Hackathon/HTN/opencv_face_detector.caffemodel'
    CONFIG_PATH = '/Users/Timot/Desktop/Hackathon/HTN/deploy.prototxt'
elif args.model == 'yoloV3':
    WEIGHTS_PATH = '/Users/Timot/Desktop/Hackathon/HTN/yolov3.caffemodel'
    CONFIG_PATH = '/Users/Timot/Desktop/Hackathon/HTN/yolov3.prototxt'


class PeopleDetector(object):
    def __init__(self,
                 weights_path: str = WEIGHTS_PATH,
                 config_path: str = CONFIG_PATH):

        if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
            raise Exception('No model pre-loaded... need to load the caffe model')

        self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
        # with open("/Users/Timot/Desktop/Hackathon/HTN/f_rcnn_rn152_800_1333.pb") as f:
        #     self.net = f.read()       

        print('model loaded..')

    def process(self, frame, conf_threshold=0.5):

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        print(detections.shape)

        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        bboxes = []
        class_id = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                xmin = int(detections[0, 0, i, 3] * frame.shape[1])
                ymin = int(detections[0, 0, i, 4] * frame.shape[0])
                xmax = int(detections[0, 0, i, 5] * frame.shape[1])
                ymax = int(detections[0, 0, i, 6] * frame.shape[0])
                bboxes.append([xmin, ymin, xmax, ymax])
                class_id.append(int(detections[0, 0, i, 1])) # Class label

        return bboxes


def run():
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

# model_spec = {
#         'order_pos': 1, 'dim_pos': 2, # position is a center in 2D space; under constant velocity model
#         'order_size': 0, 'dim_size': 2, # bounding box is 2 dimensional; under constant velocity model
#         'q_var_pos': 1000., # process noise
#         'r_var_pos': 0.1 # measurement noise
#     }

# tracker = MultiObjectTracker(dt=1 / 10, model_spec=model_spec)

    dt = 1 / 15.0  # assume 8 fps
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)
    input_video = args.input_video

    # open camera
    cap = cv2.VideoCapture(input_video)

    # vid = imageio.get_reader(input_video, 'ffmpeg')

    people_detector = PeopleDetector()

    while(True):
        ret, frame = cap.read()

        # frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # run face detector on current frame
        bboxes = people_detector.process(frame, args.confidence)
        detections = [Detection(box=bbox) for bbox in bboxes]
        logger.debug(f'detections: {detections}')

        tracker.step(detections)
        tracks = tracker.active_tracks(min_steps_alive=3)
        logger.debug(f'tracks: {tracks}')

        # preview the boxes on frame
        for det in detections:
            draw_detection(frame, det)

        for track in tracks:
            draw_track(frame, track)

        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

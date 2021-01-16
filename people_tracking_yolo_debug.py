import os
from urllib.request import urlretrieve
import cv2
from loguru import logger

from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection
import math
import imageio

# YOLO object detection
from flask import Flask, request, Response, jsonify
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io

import cv2 as cv
import numpy as np
from numpy import mean
import time
import argparse
import configparser
from PIL import Image, ImageFont, ImageDraw 
from collections import deque 

import json

"""
adapated
https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html


Combine above with people_tracking software from motby
"""


''' from motby library'''
def draw_text(img, text, above_box, color=(0, 0, 255)):
    tl_pt = (int(above_box[0]), int(above_box[1]) - 7)
    cv2.putText(img, text, tl_pt,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color)
    return img

def draw_rectangle(img, box, color, thickness: int = 3):
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
        box[2]), int(box[3])), color, thickness)
    return img

def custom_draw_track(img, track, object_ID, random_color: bool = True, fallback_color=(200, 20, 20)):
    if object_ID == 'person':
        color = [ord(c) * ord(c) % 256 for c in track.id[:3]] if random_color else fallback_color
        th = 1
    else:
        color = [255, 255, 255]
        th = 3
    img = draw_rectangle(img, track.box, color=color, thickness=th)
    img = draw_text(img, track.id, above_box=track.box)
    return img
''' for customization '''

def draw_line(img, x1, y1, x2, y2, color, th):
    print(round(x1))
    img = cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, th)
    return img

def distance_2d(x1, y1, x_lst, y_lst, img):
    dist = []
    for idx, x2 in enumerate(x_lst):
        y2 = y_lst[idx]
        dist.append(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
    index_min = min(range(len(dist)), key=dist.__getitem__)

    img = draw_line(img, x1, y1, x_lst[index_min], y_lst[index_min], [255, 255, 255], 3)

    return index_min, img

def save_video(filename, volume, image_shape):
    # https://stackoverflow.com/questions/61981413/how-do-i-save-a-list-of-arrays-frames-to-a-video-using-opencv
    out = cv2.VideoWriter(filename + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, image_shape)
    for frame in volume:
        frame = cv2.resize(frame, image_shape, interpolation = cv2.INTER_AREA)

        out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()
    return

def exists(key, my_list):
    ''' https://stackoverflow.com/questions/39641229/how-to-find-a-given-element-in-nested-lists'''
    for item in my_list:
        if isinstance(item, list):
            if exists(key, item):  # <--Recursive Call
                return True
        elif item == key:
            return True
    return False

def write_stats(img, num_people, time_stamp, update_text_font):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    title_text = 'n = ' + str(num_people) + ' ; ' + str(time_stamp)
    # org 
    org = (10, round(img.shape[1]*0.5)+80) 
    # fontScale 
    fontScale = 0.75
    # Blue color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    img = cv2.putText(img, title_text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
                    

    cv.imshow('window', img)
    return np.asarray(img)

def clean_bbox_dict(d, d_bbox):
    print(d)
    print(d_bbox)
    print('cleaning')
    input('press to continue..')
    return d, d_bbox

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-m', '--model', type=str, default='MobileNet')
parser.add_argument('-i', '--input_video', type=str, default='mall')
parser.add_argument('-c', '--confidence', type=float, default=0.5)
parser.add_argument('-sm', '--SlowMode', type=int, default=0)
parser.add_argument('-sb', '--save_bool', type=int, default=1)
parser.add_argument('-v', '--verbose', type=int, default=0)
parser.add_argument('-fb', '--flask_bool', type=int, default=1)


# run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

args = parser.parse_args()

# Initialize the Flask application
if args.flask_bool:
    app = Flask(__name__)
    # route http posts to this method

# @app.route('/api/test', methods=['GET'])
def main():
    ID_only = []

    verbose = args.verbose

    if args.model == 'yolov3':
        CONFIG_PATH, WEIGHTS_PATH = 'yolov3.cfg', 'yolov3.weights'

    if not os.path.isfile(WEIGHTS_PATH):
        logger.debug('downloading model...')
        urlretrieve('https://pjreddie.com/media/files/yolov3.weights', WEIGHTS_PATH)

    if args.input_video == 'mall':
        input_video = 'sample_mall_vid.mp4'
        fx, fy = 1, 1
        x1_loc = 140
        y1_loc = 200
        x2_loc = 340
        y2_loc = 250

    elif args.input_video == 'shop':
        input_video = 'sample_shop_vid.mp4'
        fx, fy = 0.7, 0.7
        x1_loc = 600
        y1_loc = 500
        x2_loc = 740
        y2_loc = 390

    update_text_font = ImageFont.truetype("arial.ttf", 15)

    # Load names of classes and get random colors
    classes = open('coco.names').read().strip().split('\n')

    accepted_classes = ['person', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase']
    accessory_ref_lst = ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase']
    inner_keys = ['object', 'time', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase']


    idx_accepted = [0, 24, 25, 26, 27, 28]

    np.random.seed(42)

    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('model loaded')

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # open camera
    cap = cv2.VideoCapture(input_video)
    dt = 1 / 8.0  # assume 8 fps

    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1}

    # prepare tracking
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    # python dictionary to track people
    d = {
        'ID': 
            {'object': 'value_1', 'time': 'value_2', 'backpack': 'value_3', 'umbrella': 'value_4', 'handbag': 'value_5', 'tie': 'value_6', 'suitcase': 'value_7'}
    }

    d_bbox = {'ID': 
            {'x1': 'value_1', 'y1': 'value_2', 'x2': 'value_3', 'y2': 'value_4'}
    }

    arr_d = []

    ctr = 0
    clear_ctr = 0
    img_array = []
    while(True):
            # only process every 30 frames
        if args.input_video == 'shop' and ctr < 45:
            # shop example frozen for the first 40 frames
            ret, img = cap.read()
            ctr += 1
            continue
        # while True:
        ret, img = cap.read()
        clear_ctr += 1
        # save if end of video file
        if img is None:
            if args.save_bool:
                save_video(args.input_video, img_array, size)
                # exit
                break

        img = cv2.resize(img, dsize=None, fx=fx, fy=fy)
        size = (img.shape[1], img.shape[0])

        # construct a blob from the image
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        if verbose: t0 = time.time()
        outputs = net.forward(ln)
        if verbose: t = time.time()
        if verbose: print('time=', t-t0)

        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                # ignore if not in classes we want
                if classID not in idx_accepted:
                    continue
                # logger.debug(f'class: {classes[classID]}')
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        class_lst = []

        bboxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # # old version of boxes without ID tracking
                color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                class_lst.append(classes[classIDs[i]])
                # getting the boundaries of the box for tracking
                xmin = int(x)
                ymin = int(y)
                xmax = int(x + w)
                ymax = int(y + h)
                bboxes.append([xmin, ymin, xmax, ymax])
        
        # if empty list
        if not class_lst:
            continue

        ''' detection adapated from https://learnopencv.com/goturn-deep-learning-based-object-tracking/ '''
        detections = [Detection(box=bbox) for bbox in bboxes]
        if verbose: logger.debug(f'detections: {detections}')
        
        # edited MOTPY tracker source code
        
        # tracker.step(detections, class_lst)

        tracks = tracker.active_tracks(min_steps_alive=-1)
        if verbose: logger.debug(f'tracks: {tracks}')


        print(detections)
        # prepare text for each person detected
        # text_arr = []



        # # preview the boxes on frame
        # for det in detections:
        #     draw_detection(img, det)

        # u_x_p = []
        # u_y_p = []
        # u_x_a = []
        # u_y_a = []

        # people_track_lst = []
        # accessories_track_lst = []
        # for idx, track in enumerate(tracks):
        #     bound_box = track[1]

        #     ID = track[0].split('-')[0]
        #     class_ID = track[0].split('-')[1]
            
        #     # append to sort 
        #     if class_ID == 'person':
        #         people_track_lst.append(track)
        #         u_x_p.append(mean([bound_box[0], bound_box[2]]))
        #         u_y_p.append(mean([bound_box[1], bound_box[3]]))
        #         custom_draw_track(img, track, 'person')

        #     else:
        #         accessories_track_lst.append(track)
        #         u_x_a.append(mean([bound_box[0], bound_box[2]]))
        #         u_y_a.append(mean([bound_box[1], bound_box[3]]))
        #         custom_draw_track(img, track, 'accessory')

        #     # custom_draw_track(img, track, text_arr[idx])

        # time_stamp = time.strftime("%Y%m%d%H%M%S")

        # # combine the track list, but accessories ordered last
        # track_list = people_track_lst + accessories_track_lst
        # ux = u_x_p + u_x_a
        # uy = u_y_p + u_y_a

        # # determine how many people detected
        # if len(indices) > 0:
        #     # process bag and count people
        #     for idx, track in enumerate(track_list):
        #         bound_box = track[1]

        #         ID = track[0].split('-')[0]
        #         class_ID = track[0].split('-')[1]

        #         bp_curr = None
        #         ub_curr = None
        #         hb_curr = None
        #         t_curr = None
        #         sc_curr = None
        #         status_stamp = None
        #         px_h = None

        #         # if accessory 
        #         if class_ID != 'person':
        #             # calculate a list of distances between the people and this point
        #             person_index, img = distance_2d(ux[idx], uy[idx], u_x_p, u_y_p, img)

        #             # if it was not registered as an accessory yet
        #             # if exists(ID, arr_d) is False:
        #             # Check if key exist in dictionary using any()
        #             if any(ID in d_t.values() for d_t in d.values()) is False:
        #                 print(any(ID in d_t.values() for d_t in d.values()))

        #                 # index of the person...
        #                 curr_person = people_track_lst[person_index]
        #                 owner_ID = curr_person[0].split('-')[0]

        #                 # set the new value into the dictionary
        #                 d[owner_ID][class_ID] = ID

        #         # add to dictionary (changed to list) if it doesn't exist
        #         # elif exists(ID, arr_d) is False:
        #         elif ID not in d.keys():
        #             d.update({
        #                 ID: {'object': class_ID, 'time': time_stamp, 'status': status_stamp, 'height': px_h, 'backpack': bp_curr, 'umbrella': ub_curr, 'handbag': hb_curr, 'tie': t_curr, 'suitcase': sc_curr}
        #             })

        #             d_bbox.update({
        #                 ID: {'x1': [round(bound_box[0])], 'y1': [round(bound_box[1])], 'x2': [round(bound_box[2])], 'y2': [round(bound_box[3])]}
        #             })
        #         # every two frames, we append 
        #         elif clear_ctr % 2:
        #             # it's already in the list, we append the position
        #             d_bbox[ID]['x1'].append(round(bound_box[0]))
        #             d_bbox[ID]['y1'].append(round(bound_box[1]))
        #             d_bbox[ID]['x2'].append(round(bound_box[2]))
        #             d_bbox[ID]['y2'].append(round(bound_box[3]))

        #             # arr_d.append([ID, class_ID, time_stamp, bp_curr, ub_curr, hb_curr, t_curr, sc_curr])
        #             # ID_only.append(ID)

        # # print(d_bbox)
        # # every 20 frames, we remove idle status objects from the dictionaries
        # if clear_ctr % 2:
        #     d, d_bbox = clean_bbox_dict(d, d_bbox)

        # # print(d)
        # # print(ID_only)
        # num_people = len(people_track_lst)

        # # get time stamp
        # img = write_stats(img, num_people, time_stamp, update_text_font)

        # if verbose: logger.debug(f'number of people: {num_people}, time of day: {time_stamp}')

        # # draw line for people counting
        # img = draw_line(img, x1_loc, y1_loc, x2_loc, y2_loc, (0, 0, 255), 5)

        cv.imshow('window', img)
                # stop demo by pressing 'q'
        if cv2.waitKey(int(1000*dt)) & 0xFF == ord('q'):
            break

        img_array.append(img)

        if args.SlowMode:
            input("Press Enter to continue...")
            with open('shop.json', 'w') as json_file:
                json.dump(d, json_file, indent=4)
        
        # uncomment to route!
        if args.flask_bool: return Response(response=str(d), status=200,mimetype="application/json")
            
    # start flask app
if __name__ == '__main__':
    if args.flask_bool:
        app.run(debug=True, host='0.0.0.0')
    else:
        main()
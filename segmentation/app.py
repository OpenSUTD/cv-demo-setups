# ================
# RealSense Config
# ================

import numpy as np
import cv2
import time
import pyrealsense2 as rs
from threading import Thread

class Camera():
    def __init__(self):
        self.colour_frame = None
        self.depth_frame = None
        self.stopped = False

    def start(self):
        print("Starting a new thread to stream frames from connected RealSense camera")
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 6)

        profile = config.resolve(pipeline)

        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        align_to = rs.stream.color
        align = rs.align(align_to)

        #time.sleep(1.0)

        # keep looping infinitely until the thread is stopped
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.06), cv2.COLORMAP_RAINBOW)
            depth_colormap = cv2.resize(depth_colormap, (320,180))
            depth_colormap = cv2.flip(depth_colormap, 1)

            color_image = np.asanyarray(color_frame.get_data())
            color_image_f = cv2.flip(color_image, 1)

            self.colour_frame = color_image_f
            self.depth_frame = depth_colormap

            time.sleep(1/7)

            if self.stopped:
                pipeline.stop()
                return

    def read(self):
        return self.colour_frame, self.depth_frame

    def stop(self):
        self.stopped = True

capture = Camera().start()

# =================
# Mask R-CNN Config
# =================

import os
import sys

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

import mrcnn.model as modellib
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 640

config = InferenceConfig()
config.display()

import tensorflow as tf

with tf.device("/gpu:0"):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'bag', 'umbrella', 'bag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'table', 'toilet', 'monitor', 'laptop', 'mouse', 'remote',
               'keyboard', 'phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# ==========
# Event Loop
# ==========

import functools
import colorsys

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

N = 12
hsv = [(i / N, 1, 1.0) for i in range(N)]
colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

alpha = 0.5

from numba import jit

@jit(nopython=False)
def render_masks(image, boxes, masks, class_ids, N):
    for i in range(N):
        if not np.any(boxes[i]):
            continue
        color = colors[i]
        y1, x1, y2, x2 = boxes[i]
        cent_x, cent_y = int((x1+x2)/2), int((y1+y2)/2)
        caption = class_names[class_ids[i]]
        mask = masks[:, :, i]

        image[:, :, 0] = np.where(mask == 1,
                                  image[:, :, 0] *
                                  (1 - alpha) + alpha * color[0] * 255,
                                  image[:, :, 0])
        image[:, :, 1] = np.where(mask == 1,
                                    image[:, :, 1] *
                                    (1 - alpha) + alpha * color[1] * 255,
                                    image[:, :, 1])
        image[:, :, 2] = np.where(mask == 1,
                                    image[:, :, 2] *
                                    (1 - alpha) + alpha * color[2] * 255,
                                    image[:, :, 2])

        cv2.putText(image,caption,(cent_x, cent_y), font, 0.8,(60,60,255), 2, cv2.LINE_AA)

    return image


while True:
    #start = time.time()

    color_image_f, depth_colormap = capture.read()

    r = model.detect([color_image_f])[0]

    boxes, masks = r["rois"], r["masks"]
    class_ids = r["class_ids"]
    N = boxes.shape[0]
    if N>12:
        N=12

    if not N:
        masked_image = color_image_f
    else:
        #assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        input_image = color_image_f
        masked_image = render_masks(color_image_f, boxes, masks, class_ids, N)

    masked_image = cv2.resize(masked_image, (1920,1080), interpolation=cv2.INTER_CUBIC)

    cv2.putText(depth_colormap,'DEPTH',(8, 40), font, 1.4,(255,255,255), 3, cv2.LINE_AA)

    masked_image[0:depth_colormap.shape[0], 0:depth_colormap.shape[1]] = depth_colormap

    #frame_time = time.time() - start
    #fps = round(1/frame_time,2)
    #cv2.putText(masked_image,str(fps),(1280, 80), font, 2,(0,0,255), 4, cv2.LINE_AA)

    cv2.imshow('window',masked_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.stop()


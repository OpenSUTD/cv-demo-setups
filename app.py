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
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 15)

        profile = config.resolve(pipeline)

        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        align_to = rs.stream.color
        align = rs.align(align_to)

        # keep looping infinitely until the thread is stopped
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap = cv2.resize(depth_colormap, (320,180))
            depth_colormap = cv2.flip(depth_colormap, 1)

            color_image = np.asanyarray(color_frame.get_data())
            color_image_f = cv2.flip(color_image, 1)
            self.colour_frame = color_image_f

            self.depth_frame = depth_colormap

            time.sleep(0.1)

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
    IMAGE_MIN_DIM = 360
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
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# ==========
# Event Loop
# ==========

import functools
import colorsys
import random

@functools.lru_cache(maxsize=32, typed=True)
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:

    color_image_f, depth_colormap = capture.read()

    r = model.detect([color_image_f])[0]

    boxes, masks = r["rois"], r["masks"]
    class_ids = r["class_ids"]
    N = boxes.shape[0]

    if not N:
        masked_image = color_image_f.copy()
    else:
        #assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        input_image = color_image_f.copy()
        colors = random_colors(N)
        for i in range(N):
            if not np.any(boxes[i]):
                continue
            color = colors[i]
            y1, x1, y2, x2 = boxes[i]
            cent_x, cent_y = int((x1+x2)/2), int((y1+y2)/2)
            caption = class_names[class_ids[i]]
            mask = masks[:, :, i]
            masked_image = apply_mask(input_image, mask, color)
            cv2.putText(masked_image,caption,(cent_x, cent_y), font, 1,(255,255,255), 2, cv2.LINE_AA)

    masked_image = cv2.resize(masked_image, (1920,1080))

    cv2.putText(depth_colormap,'DEPTH IMAGE',(4, 30), font, 1,(255,255,255), 2, cv2.LINE_AA)

    masked_image[0:depth_colormap.shape[0], 0:depth_colormap.shape[1]] = depth_colormap

    cv2.imshow('window',masked_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.stop()


import numpy as np
import cv2
import time
import pyrealsense2 as rs

# ================
# RealSense Config
# ================

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

profile = config.resolve(pipeline)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# =================
# Mask R-CNN Config
# =================

import os
import sys

import matplotlib
matplotlib.use('TkAgg') 
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

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
    IMAGE_MIN_DIM = 480

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

font = cv2.FONT_HERSHEY_SIMPLEX

fig = Figure(figsize=(16,9),dpi=50)
canvas = FigureCanvas(fig)
ax = fig.gca()

while True:

    ax.cla()

    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_colormap = cv2.resize(depth_colormap, (320,))
    depth_colormap = cv2.flip(depth_colormap, 1)

    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)

    r = model.detect([color_image])[0]
    visualize.display_instances(color_image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax,
                                title="Predictions")

    canvas.draw()
    
    buffer = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
    mask_image = buffer.reshape(canvas.get_width_height()[::-1] + (3,))

    final_image = cv2.resize(mask_image, (1920,1080))

    final_image[0:depth_colormap.shape[0], 0:depth_colormap.shape[1]] = depth_colormap

    cv2.putText(final_image,'DEPTH IMAGE',(2, 30), font, 1,(255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('frame',final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
pipeline.stop()
cv2.destroyAllWindows()

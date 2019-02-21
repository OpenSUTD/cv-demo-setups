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
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)
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
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_RAINBOW)
            depth_colormap = cv2.resize(depth_colormap, (320,180))
            depth_colormap = cv2.flip(depth_colormap, 1)

            color_image = np.asanyarray(color_frame.get_data())
            color_image_f = cv2.flip(color_image, 1)

            self.colour_frame = color_image_f
            self.depth_frame = depth_colormap

            time.sleep(0.05)

            if self.stopped:
                pipeline.stop()
                return

    def read(self):
        return self.colour_frame, self.depth_frame

    def stop(self):
        self.stopped = True

from collections import deque
import imutils

capture = Camera().start()

BUFFER = 64

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (30, 60, 20)
greenUpper = (60, 255, 255)
pts = deque(maxlen=BUFFER)

import sys

sys.path.append('./openpose/build/python')

from openpose import pyopenpose as op

def set_params():
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["model_folder"] = "./openpose/models"
        return params

params = set_params()

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# keep looping
while True:
	# grab the current frame
	raw_image, depth_colormap = capture.read()

	frame = cv2.resize(raw_image, (640, 360))

	datum = op.Datum()
	datum.cvInputData = frame
	opWrapper.emplaceAndPop([datum])
	pose = datum.cvOutputData
	pose_frame = frame + pose

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# Display the stream

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(BUFFER / float(i + 1)) * 2.5)
		cv2.line(pose_frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		
	output_image = cv2.resize(pose_frame, (1920,1080), interpolation=cv2.INTER_CUBIC)
	cv2.imshow("window", output_image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# close all windows

cv2.destroyAllWindows()
capture.stop()

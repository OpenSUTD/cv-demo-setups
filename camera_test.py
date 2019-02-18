import numpy as np
import cv2
import time

import pyrealsense2 as rs

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = config.resolve(pipeline)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_colormap = cv2.resize(depth_colormap, (320,160))
    
    depth_colormap = cv2.flip(depth_colormap, 1)

    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)

    final_image = color_image

    x_offset, y_offset = 0, 0
    final_image[y_offset:y_offset+depth_colormap.shape[0], x_offset:x_offset+depth_colormap.shape[1]] = depth_colormap

    cv2.putText(final_image,'DEPTH IMAGE',(2, 30), font, 1,(255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('frame',final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

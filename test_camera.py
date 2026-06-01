import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
camera = Camera(crop_mode=1)

while True:
    image = camera.get_image()
    if image is None:
        continue

    cv2.imshow("Color", image)

    if cv2.waitKey(1) == ord('q'):
        break
camera.close()
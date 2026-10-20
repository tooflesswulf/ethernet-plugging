import time
import numpy as np
import pyrealsense2 as rs


class Camera:
    """
    RealSense RGB camera wrapper.

    crop_mode:
        -1 -> left crop
         0 -> center crop
         1 -> right crop
    """

    def __init__(
        self,
        sid="843212070496",
        width=640,
        height=480,
        fps=30,
        crop_mode=0,
    ):
        self.crop_mode = crop_mode

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Connect to specific camera
        self.config.enable_device(sid)

        # Enable RGB stream
        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )

        # Start pipeline
        self.pipeline.start(self.config)

        # Warm-up
        time.sleep(1.0)

    def _square_crop(self, image):
        """
        Apply square crop based on crop_mode.
        """
        h, w = image.shape[:2]
        size = min(h, w)

        # Already square
        if h == w:
            return image

        # Wider than tall
        if w > h:
            if self.crop_mode == -1:
                x0 = 0
            elif self.crop_mode == 1:
                x0 = w - size
            else:
                x0 = (w - size) // 2

            return image[:, x0:x0 + size]

        # Taller than wide
        else:
            y0 = (h - size) // 2
            return image[y0:y0 + size, :]

    def get_image(self):
        """
        Returns:
            np.ndarray: (H, W, 3) uint8 BGR image
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to get color frame from camera")

        image = np.asanyarray(color_frame.get_data())
        image = self._square_crop(image)
        return image

    def close(self):
        self.pipeline.stop()

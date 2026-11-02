"""
RealSense RGB camera, in-process (Camera) or in a child process (CameraProcess).

In the robot process, the camera thread's wait_for_frames shares one GIL with the 500 Hz
control loop and every other thread. In logs-video-75 (2026-10-30) the whole process froze
for 50-280 ms every 10-20 s, every thread at once, and 86% of the freezes ended within
5 ms of a camera frame arriving: a frame that comes late holds everyone up. CameraProcess
waits for frames in a child with its own interpreter; the parent's camera thread only
waits on a pipe, which releases the GIL.
"""
import multiprocessing as mp
import signal
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


def _serve(conn, kwargs):
    # Ctrl-C reaches the whole process group: let the parent stop us through close()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        camera = Camera(**kwargs)
    except Exception as e:
        conn.send(RuntimeError(f'camera: {e}'))
        return
    conn.send('ready')
    try:
        while conn.recv():
            try:
                conn.send(np.ascontiguousarray(camera.get_image()))
            except RuntimeError as e:       # e.g. no frame within wait_for_frames' timeout
                conn.send(RuntimeError(f'camera: {e}'))
    except EOFError:                        # the parent went away
        pass
    finally:
        camera.close()


class CameraProcess:
    """
    Camera(**kwargs) in a child process, with the same get_image() / close(). Requests are
    synchronous, one frame per get_image(), so frames are as fresh as Camera's.
    """

    def __init__(self, **kwargs):
        ctx = mp.get_context('spawn')
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, kwargs), daemon=True)
        self._proc.start()
        child.close()
        msg = self._conn.recv()             # once the pipeline is up and warmed up
        if isinstance(msg, Exception):
            self._proc.join()
            raise msg

    def get_image(self):
        """
        Returns:
            np.ndarray: (H, W, 3) uint8 BGR image (a copy, owned by the caller)
        """
        self._conn.send(True)
        msg = self._conn.recv()
        if isinstance(msg, Exception):
            raise msg
        return msg

    def close(self):
        if self._proc.is_alive():
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=6)      # past wait_for_frames' 5 s timeout
            if self._proc.is_alive():
                self._proc.terminate()
        self._conn.close()

import logging
import sys

import cv2

try:
    sys.path.append("..")
    import PlatformConfig as Config
except ImportError:
    print("Could not import PlatformConfig from previous directory.")
    raise SystemExit


class CameraManager:
    def __init__(self, camera_index=Config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.status = False
        self.logger = logging.getLogger("epic2023.CameraManager")

    def open_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.status = True
                self.logger.info(f"Camera[{self.camera_index}] opened")
                return True
            else:
                self.status = False
                self.logger.error(f"Failed to open camera[{self.camera_index}]")
        except Exception as e:
            self.status = False
            self.logger.error(f"Error opening camera: {e}")

    def close_camera(self):
        self.cap.release()
        self.status = False
        self.logger.info(f"Camera[{self.camera_index}] closed.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

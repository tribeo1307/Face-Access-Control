# Camera management module
import cv2
import time

class Camera:
    def __init__(self, camera_id=0, target_fps=30):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.cap = None
        self.frame_delay = 1.0 / target_fps if target_fps > 0 else 0

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Không thể mở camera ID {self.camera_id}")
        
        print(f"Camera ID {self.camera_id} đã được mở.")
        return self

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            print(f"Camera ID {self.camera_id} đã được giải phóng.")
        
        return False

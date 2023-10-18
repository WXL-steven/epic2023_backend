import logging
import sys
import time

import cv2
from PIL import Image

sys.path.append(".")
try:
    import PlatformConfig as Config
    from Packages.Serial import SerialManager
    from Packages.WebSocketManager import WebSocketServer
    from Packages.NeuralNetwork import ImageClassifier
    from Packages.RequestParser import SerialParser, WSParser
    from Packages.Camera import CameraManager
except ImportError:
    print("Could not import PlatformConfig.")
    raise SystemExit


class ClassifierSignal:
    signal = False

    @staticmethod
    def activate():
        ClassifierSignal.signal = True

    @staticmethod
    def deactivate():
        ClassifierSignal.signal = False


def main():
    """
    The main function of the program.
    """
    logger = logging.getLogger("epic2023")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    serial = SerialManager()
    server = WebSocketServer(22334)

    serial.callback_init(server.send_messages, ClassifierSignal.activate)
    server.callback_init(serial.write_to_port)
    server.start_server()
    serial.open_port()
    server.add_connect_callback(server.send_device_status, "mcu", "ready")

    cf = ImageClassifier()

    camera = CameraManager()
    if camera.open_camera() and camera.status:
        if camera.get_frame() is not None:
            server.send_device_status("camera", "ready")
            server.add_connect_callback(server.send_device_status, "camera", "ready")
        else:
            server.send_device_status("camera", "error")
            server.add_connect_callback(server.send_device_status, "camera", "error")
    else:
        server.send_device_status("camera", "error")
        server.add_connect_callback(server.send_device_status, "camera", "error")

    try:
        last_shoot_time = time.time()
        while True:
            if ClassifierSignal.signal:
                frame = camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if frame is not None:
                    result = cf.predict(Image.fromarray(frame))
                    serial.send_classifier_result(result)
                    ClassifierSignal.deactivate()
            elif time.time() - last_shoot_time > 1/Config.DISPLAY_FRAME_RATE:
                frame = camera.get_frame()
                if frame is not None:
                    server.send_image(frame, "realtimeImageTransfer")
                    last_shoot_time = time.time()
            else:
                time.sleep(0.001)
            # time.sleep(0.001)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        camera.close_camera()
        serial.close_port()
        server.stop_server()


if __name__ == "__main__":
    main()

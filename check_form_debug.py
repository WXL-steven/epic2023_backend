import logging
import random
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


def message_send_emulator(message):
    print(message)


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

    serial = SerialManager(port="COM3")

    serial.callback_init(message_send_emulator, ClassifierSignal.activate)
    serial.open_port()

    cf = ImageClassifier()

    try:
        img_path = [r'./h.png', r'./k.png', r'./o.png', r'./r.png']
        while True:
            if ClassifierSignal.signal:
                frame = cv2.imread(random.choice(img_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if frame is not None:
                    result = cf.predict(Image.fromarray(frame))
                    serial.send_classifier_result(result)
                    ClassifierSignal.deactivate()
            else:
                time.sleep(0.005)
            # time.sleep(0.001)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        serial.close_port()


if __name__ == "__main__":
    main()

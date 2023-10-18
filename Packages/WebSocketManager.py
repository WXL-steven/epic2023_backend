import base64
import json
import sys
import threading
import time
import logging
from queue import Queue

import cv2
import tornado.ioloop
import tornado.websocket
import weakref

try:
    sys.path.append("..")
    import PlatformConfig as Config
    from Packages.RequestParser import SerialParser, WSParser
except ImportError:
    print("Could not import Module.")
    raise SystemExit


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    connections = weakref.WeakSet()

    def initialize(self, wsparser, wshandler, connect_callback):
        self.parser = wsparser
        self.handler = wshandler
        self.connect_callback = connect_callback

    def open(self):
        self.connections.add(self)
        self.logger = logging.getLogger(f"epic2023.WebSocketHandler.{self.request.remote_ip.replace('.', '_')}")
        self.logger.info(f"New client connected from {self.request.remote_ip}.")
        task_list = list(self.connect_callback.queue)
        for task_bundle in task_list:
            if task_bundle is not None:
                task, args, kwargs = task_bundle
                task(*args, **kwargs)

    def on_message(self, message):
        command = self.parser.package_parser(message)
        self.handler(command)

    def on_close(self):
        self.connections.remove(self)
        self.logger.info(f"Client disconnected. Total connected clients: {len(self.connections)}")

    @classmethod
    def close_all_connections(cls):
        for connection in list(cls.connections):
            connection.close()


class WebSocketServer:
    def __init__(self, port):
        self.port = port
        self.parser = WSParser()
        self.handler = None
        self.running = True
        self.loop = None
        self.thread = None
        self.connect_callback = Queue()
        self.logger = logging.getLogger("epic2023.WebSocketServer")

    def callback_init(self, handler):
        self.handler = handler

    def add_connect_callback(self, callback, *args, **kwargs):
        self.connect_callback.put((callback, args, kwargs))

    def clear_connect_callback(self):
        self.connect_callback = Queue()

    def start_server(self):
        if self.handler is None:
            self.logger.error("Could not start WebSocket server: handler is not initialized.")
            return

        def start_loop():
            app = tornado.web.Application(
                [(r"/",
                  WebSocketHandler,
                  dict(wsparser=self.parser, wshandler=self.handler, connect_callback=self.connect_callback))]
            )
            app.listen(self.port)
            self.loop = tornado.ioloop.IOLoop.current()
            self.loop.add_callback(lambda: self.logger.info(f"WebSocket server started on port {self.port}."))
            self.loop.start()
            self.logger.info("WebSocket server loop stopped.")

        self.thread = threading.Thread(target=start_loop)
        self.thread.start()

    def stop_server(self):
        self.running = False
        WebSocketHandler.close_all_connections()
        self.loop.add_callback(self.loop.stop)
        self.thread.join()

    @staticmethod
    def _send_message_thread_safety(message):
        if len(WebSocketHandler.connections) == 0:
            return
        for connection in list(WebSocketHandler.connections):
            connection.write_message(message)

    def send_messages(self, message):
        if message is None:
            return
        if not isinstance(message, str):
            self.logger.error(f"Could not send message: message({type(message)}) is not a string.")
            return  # Only send string messages
        self.loop.add_callback(self._send_message_thread_safety, message)

    def send_image(self, image, service="resultImageTransfer", chunk_size=8192):
        if len(WebSocketHandler.connections) == 0:
            return
        retval, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        jpg_as_text = base64.b64encode(buffer)
        chunks = [jpg_as_text[i:i + chunk_size] for i in range(0, len(jpg_as_text), chunk_size)]
        timestamp = int(time.time() * 1000)
        for chunk_index, chunk in enumerate(chunks):
            message = json.dumps({
                "application": "epic2023",
                "service": service,
                "totalChunks": len(chunks),
                "chunkIndex": chunk_index,
                "timestamp": timestamp,
                "base64Data": chunk.decode('utf-8')  # Decode bytes to string
            })
            self.send_messages(message)

    def send_device_status(self, device_name, status):
        if len(WebSocketHandler.connections) == 0:
            return
        message = json.dumps({
            "application": "epic2023",
            "service": "deviceStatusManager",
            "deviceName": device_name,
            "deviceStatus": status
        })
        self.logger.info(f"Sending device status: {device_name} - {status}")
        self.send_messages(message)

    def add_trash(self, trash_type, count=1):
        if len(WebSocketHandler.connections) == 0:
            return
        message = json.dumps({
            "application": "epic2023",
            "service": "updateTrashCount",
            "trashType": trash_type.lower(),
            "trashCount": count
        })
        self.send_messages(message)


if __name__ == "__main__":
    logger = logging.getLogger("epic2023")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    server = WebSocketServer(22335)
    server.start_server()

    fn = [r'../h.png', r'../k.png', r'../o.png', r'../r.png']
    try:
        while True:
            time.sleep(1)
            server.send_messages("Hello world!")
    except KeyboardInterrupt:
        server.stop_server()

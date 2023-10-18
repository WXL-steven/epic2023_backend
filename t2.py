import tornado.ioloop
import tornado.websocket
import threading
import time
import weakref


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    connections = weakref.WeakSet()

    def open(self):
        self.connections.add(self)

    def on_close(self):
        self.connections.remove(self)

    # @classmethod
    # def send_to_all(cls, message):
    #     for connection in cls.connections:
    #         connection.write_message(message)


def start_server():
    app = tornado.web.Application([(r"/", WebSocketHandler)])
    app.listen(22334)
    tornado.ioloop.IOLoop.current().start()


def send_periodic_message():
    while True:
        for connection in WebSocketHandler.connections:
            connection.write_message("Hello world!")
        # WebSocketHandler.send_to_all("Hello world!")
        time.sleep(1)  # broadcast "Hello world!" every second


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    message_thread = threading.Thread(target=send_periodic_message)
    message_thread.start()

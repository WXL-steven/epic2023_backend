import logging

# 串口配置
SERIAL_PORT = '/dev/ttyS3'
SERIAL_BAUD = 115200
SERIAL_BUFFER_SIZE = 1024
SERIAL_PACKAGE_TIMEOUT = 1

# WebSocket服务器配置
# REQUEST_BUFFER_SIZE = 1024
DISPLAY_FRAME_RATE = 30

# 摄像头配置
CAMERA_INDEX = 0

# 日志配置
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

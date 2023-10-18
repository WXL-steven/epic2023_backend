import logging
import queue
import sys
import threading
import time
from collections import deque

import serial

try:
    sys.path.append("..")
    import PlatformConfig as Config
    from Packages.RequestParser import SerialParser, WSParser
except ImportError:
    print("Could not import PlatformConfig from previous directory.")
    raise SystemExit


class CustomExecutor:
    def __init__(self):
        self.periodic_tasks = queue.Queue()
        self.temp_tasks = queue.Queue()
        self.should_stop = False
        self.worker_thread = threading.Thread(target=self._worker)

    def start(self):
        self.worker_thread.start()

    def stop(self):
        self.should_stop = True

    def add_periodic_task(self, task, *args, **kwargs):
        self.periodic_tasks.put((task, args, kwargs))

    def clear_periodic_tasks(self):
        self.periodic_tasks = queue.Queue()

    def add_temp_task(self, task, *args, **kwargs):
        self.temp_tasks.put((task, args, kwargs))

    def clear_temp_tasks(self):
        self.temp_tasks = queue.Queue()

    def _worker(self):
        while not self.should_stop:
            # Try to get a temp task first
            try:
                task, args, kwargs = self.temp_tasks.get_nowait()
                task(*args, **kwargs)
                continue
            except queue.Empty:
                pass

            # If no temp task, try to get a periodic task
            try:
                task, args, kwargs = self.periodic_tasks.get_nowait()
                task(*args, **kwargs)
                self.periodic_tasks.put((task, args, kwargs))  # Re-add the task to the queue
            except queue.Empty:
                time.sleep(0.01)  # Sleep if there are no tasks


class SerialManager:
    """
    A manager class for handling bidirectional communication over a serial port.

    Attributes:
        ser (serial.Serial): The serial port interface.
        executor (CustomExecutor): A thread pool executor for running tasks.
        buffer (deque): A deque used as a software buffer for the incoming data.
        stop_signal (bool): A signal to stop the reception thread.
    """

    def __init__(self, port=Config.SERIAL_PORT, baudrate=Config.SERIAL_BAUD, timeout=0):
        """
        Initializes the SerialManager with a port, baudrate, and timeout.

        Args:
            port (str): The serial port to connect to.
            baudrate (int): The baudrate for the serial communication.
            timeout (float): Read timeout in seconds.
        """
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.timeout = timeout
        self.buffer = deque(maxlen=Config.SERIAL_BUFFER_SIZE)
        self.executor = CustomExecutor()
        self.stop_signal = False
        self.recording = False
        self.start_time = None
        self.logger = logging.getLogger("epic2023.SerialManager")
        self.parser = None
        self.courier = None

    def callback_init(self, courier, call_classifier=None):
        self.parser = SerialParser(call_classifier)
        self.courier = courier

    def open_port(self):
        """
        Opens the serial port and starts the reception thread.

        Raises:
            SerialException: If the port cannot be opened.
        """
        if not self.ser.is_open:
            try:
                self.ser.open()
                self.stop_signal = False
                self.executor.add_periodic_task(self.read_from_port)
                self.executor.start()
                self.logger.info(f"Serial port opened: {self.ser.name}")
            except serial.SerialException as e:
                self.logger.error(f"Could not open serial port: {e}")

    def close_port(self):
        """
        Closes the serial port.
        """
        if self.ser.is_open:
            # Submit a task to set stop_signal and close the port
            self.executor.add_temp_task(self._close_port_task)

    def _close_port_task(self):
        # Set stop_signal to True so that read_from_port will not be resubmitted
        self.stop_signal = True

        # It's now safe to close the port
        self.ser.close()

        self.logger.info("Serial port closed.")

        # Wait for all tasks to complete before closing the port
        self.executor.stop()

    def read_from_port(self):
        """
        Reads data from the serial port one byte at a time and stores it in the buffer.

        Raises:
            SerialException: If data cannot be read from the port.
        """
        try:
            data = self.ser.read(1)  # read one byte
            if data is None or len(data) == 0:
                return

            if data == b'\x02':
                self.recording = True
                self.start_time = time.time()
                self.buffer.clear()
            elif data == b'\x03':
                self.recording = False
                data_package_str = b''.join(self.buffer).decode('utf-8', 'ignore')
                if self.parser is not None and self.courier is not None:
                    self.logger.info(f"Received data package: {data_package_str}")
                    command = self.parser.package_parser(data_package_str)
                    self.courier(command)
                else:
                    self.logger.error("Callback not initialized.")
                self.buffer.clear()
            elif self.recording:
                self.buffer.append(data)

            # Check for timeout
            if self.recording and (time.time() - self.start_time) > Config.SERIAL_PACKAGE_TIMEOUT:
                self.logger.error("Package timeout, discarding current buffer.")
                self.buffer.clear()
                self.recording = False

        except serial.SerialException as e:
            self.logger.error(f"Could not read from serial port: {e}")

    # DEBUG ONLY
    def debug_parser(self, data):
        if self.parser is not None and self.courier is not None:
            command = self.parser.package_parser(data)
            self.courier(command)
        else:
            self.logger.error("Callback not initialized.")

    def _write_to_port_task(self, data):
        """
        Task to write data to the serial port.

        Args:
            data (bytes): The data to send.

        Raises:
            SerialException: If data cannot be written to the port.
        """
        try:
            self.ser.write(data)
        except serial.SerialException as e:
            self.logger.error(f"Could not write to serial port: {e}")

    def write_to_port(self, data):
        """
        Writes data to the serial port.

        Args:
            data (bytes): The data to send.

        Raises:
            SerialException: If data cannot be written to the port.
        """
        if self.ser.is_open and data is not None:
            self.logger.info(f"Writing to serial port: {data}")
            self.executor.add_temp_task(self._write_to_port_task, data)
        elif not self.ser.is_open:
            self.logger.error("Could not write to serial port: port is not open.")

    def send_classifier_result(self, result):
        """
        Sends the result of the classifier to the serial port.

        Args:
            result: The result of the classifier.

        Raises:
            SerialException: If data cannot be written to the port.
        """
        if self.ser.is_open and result is not None:
            result = result[0]["label"][0].upper()
            result = result.encode('utf-8')
            self.logger.info(f"Sending classifier result to serial port: {result}")
            self.executor.add_temp_task(self._write_to_port_task, b'\x02' + b'R' + result + b'\x03')
        elif not self.ser.is_open:
            self.logger.error("Could not write to serial port: port is not open.")


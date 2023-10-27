import json
import time
import logging


class SerialParser:
    """
        This class is designed to parse incoming data packages from different devices and modules in a system.
        It translates the short-form codes into readable terms and formats them into a dictionary for further processing.

        Each parsing method is responsible for a specific type of data package. The specific type and format of the
        data package is determined by the first character of the package string.

        The class also provides a method to call a classifier, which can be used to perform additional data processing
        or decision-making based on the parsed data.

        Attributes:
            logger (Logger): Instance of the logging class to log information and debugging details.
            call_classifier (function): Optional function to call a classifier for further data processing.
    """
    _DEVICE_NAME_MAP = {
        'M': 'mcu',
        'B': 'conveyorBelt',
        'T': 'turntable',
        'P': 'tiltingPlate',
        'C': 'compressor',
        'W': 'weighing',
        'L': 'metering',
    }

    _DEVICE_STATUS_MAP = {
        'R': 'ready',
        'O': 'offline',
        'E': 'error',
    }

    _MODULE_NAME_MAP = {
        'B': 'Conveyor',
        'P': 'Compactor',
    }

    _WORK_STATUS_MAP = {
        'I': 'idle',
        'B': 'working',
    }

    _CONTAINER_NAME_MAP = {
        'H': 'hazardous',
        'R': 'recyclable',
        'K': 'kitchen',
        'O': 'other',
    }

    def __init__(self, call_classifier=None):
        """
        Constructs all the necessary attributes for the Parser object.

        Args:
            call_classifier (function, optional): Optional function to call a classifier for further data processing.
        """
        self.logger = logging.getLogger("epic2023.MessageParser")
        self.call_classifier = call_classifier

    def update_device_status(self, package):
        """
        Parses a device status update package. The package is expected to have 3 characters.
        The first character is ignored as it is the service identifier. The second character represents the device
        and the third character represents the status of the device.

        Args:
            package (str): The data package to parse.

        Returns:
            result (dict): Dictionary containing the parsed data, or None if package is invalid.
        """
        if len(package) != 3:
            self.logger.error(f"Package length is not 3. Received package: {package}")
            return None

        result = {'service': 'deviceStatusManager'}

        if package[1].upper() not in SerialParser._DEVICE_NAME_MAP:
            self.logger.error(f"Invalid device name in package: {package}")
            return None
        result['deviceName'] = SerialParser._DEVICE_NAME_MAP[package[1]]

        if package[2].upper() not in SerialParser._DEVICE_STATUS_MAP:
            self.logger.error(f"Invalid device status in package: {package}")
            return None
        result['deviceStatus'] = SerialParser._DEVICE_STATUS_MAP[package[2]]

        self.logger.info(f"Update device status: {result['deviceName']} {result['deviceStatus']}")
        return result

    def update_total_mass(self, package):
        """
        Parses a total mass update package. The package is expected to have at least 4 characters.
        The first character is ignored as it is the service identifier. The rest of the string is interpreted as a
        floating-point number representing the total mass.

        Args:
            package (str): The data package to parse.

        Returns:
            result (dict): Dictionary containing the parsed data, or None if package is invalid.
        """
        if len(package) < 4:
            self.logger.error(f"Package length is less than 4. Received package: {package}")
            return None

        result = {'service': 'updateTotalWeight'}

        try:
            result['value'] = float(package[1:])
        except ValueError:
            self.logger.error(f"Invalid value in package: {package}")
            return None

        self.logger.info(f"Update total mass: {result['value']}")
        return result

    def update_container_load(self, package):
        """
        Parses a container load update package. The package is expected to have at least 4 characters.
        The first character is ignored as it is the service identifier. The second character represents the container
        type and the rest of the string is interpreted as a floating-point number representing the load.

        Args:
            package (str): The data package to parse.

        Returns:
            result (dict): Dictionary containing the parsed data, or None if package is invalid.
        """
        if len(package) < 4:
            return None

        result = {'service': 'containerLoadManager'}

        if package[1].upper() not in SerialParser._CONTAINER_NAME_MAP:
            self.logger.error(f"Invalid container name in package: {package}")
            return None
        result['containerName'] = SerialParser._CONTAINER_NAME_MAP[package[1]]

        try:
            result['value'] = float(package[2:])
        except ValueError:
            self.logger.error(f"Invalid value in package: {package}")
            return None

        result['containerName'] = SerialParser._CONTAINER_NAME_MAP[package[1]]
        return result

    def update_work_status(self, package):
        """
        Parses a work status update package. The package is expected to have 3 characters.
        The first character is ignored as it is the service identifier. The second character represents the module
        and the third character represents the work status.

        Args:
            package (str): The data package to parse.

        Returns:
            result (dict): Dictionary containing the parsed data, or None if package is invalid.
        """
        if len(package) != 3:
            return None

        result = {'service': 'workStatusManager'}

        if package[1].upper() not in SerialParser._MODULE_NAME_MAP:
            self.logger.error(f"Invalid module name in package: {package}")
            return None
        result['moduleName'] = SerialParser._MODULE_NAME_MAP[package[1]]

        if package[2].upper() not in SerialParser._WORK_STATUS_MAP:
            self.logger.error(f"Invalid work status in package: {package}")
            return None
        result['workStatus'] = SerialParser._WORK_STATUS_MAP[package[2]]

        self.logger.info(f"Update work status: {result['moduleName']} {result['workStatus']}")
        return result

    def call_classifier(self, _):
        """
        Calls the classifier function provided during initialization.

        Args:
            _ (str): This argument is not used in this method but is required to maintain a consistent API.

        Returns:
            None
        """
        self.logger.info("Call classifier")
        if self.call_classifier is not None:
            self.call_classifier()
        return None

    _SERVICE_PARSER_MAP = {
        'S': update_device_status,
        'M': update_total_mass,
        'L': update_container_load,
        'U': update_work_status,
        'C': call_classifier,
    }

    def package_parser(self, package):
        """
        Entry point for parsing a data package. The package's first character determines the type of package and
        thus the appropriate parser method to call.

        The parsed data is then added to a dictionary with additional metadata, converted to JSON,
        and returned as a string.

        Args:
            package (str): The data package to parse.

        Returns:
            request (str): JSON-formatted string containing the request, or None if package is invalid.
        """
        if len(package) < 1:
            self.logger.error("Package length is less than 1")
            return None

        if package[0].upper() not in SerialParser._SERVICE_PARSER_MAP:
            self.logger.error(f"Cannot find service for package: {package}")
            return None

        request = SerialParser._SERVICE_PARSER_MAP[package[0]](self, package)
        if request is None:
            return None
        request['timestamp'] = int(time.time())
        request['application'] = 'epic2023'

        try:
            request = json.dumps(request)
        except TypeError:
            self.logger.error(f"Cannot dump request to JSON. Request: {request}")
            return None

        return request


class WSParser:
    """
    Parses data packages and returns a dictionary containing the parsed data.
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the Parser object.
        """
        self.logger = logging.getLogger("epic2023.WSParser")

    def request_compress(self, package):
        """
        Parses a request compress package and returns a dictionary containing the parsed data.

        Args:
            package (str): The data package to parse.

        Returns:
            result (dict): Dictionary containing the parsed data.
        """
        # if len(package) != 2:
        #     self.logger.error(f"Package length is not 2. Received package: {package}")
        #     return None

        result = b'\x02P\x03'

        return result

    _SERVICE_PARSER_MAP = {
        'C': request_compress,
    }

    def package_parser(self, package):
        """
        Parses a data package and returns a dictionary containing the parsed data.

        Args:
            package (str): The data package to parse.

        Returns:
            request (str): JSON-formatted string containing the request.
        """
        try:
            package = json.loads(package)
        except json.JSONDecodeError:
            self.logger.error(f"Could not decode package: {package}")
            return None

        if 'application' not in package:
            self.logger.error(f"Package does not contain application field: {package}")
            return None

        if package['application'] != 'epic2023':
            self.logger.error(f"Package application field is not epic2023: {package}")
            return None

        if 'service' not in package:
            self.logger.error(f"Package does not contain service field: {package}")
            return None

        if package['service'] not in WSParser._SERVICE_PARSER_MAP:
            self.logger.error(f"Package service field is not recognized: {package}")
            return None

        request = WSParser._SERVICE_PARSER_MAP[package['service']](self, package)

        return request

import logging
import atexit
from _core.config import config


class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger("deep-research-logger")
        self._setup()

    def _setup(self):
        if self.logger.handlers:  # Avoid duplicate handlers
            return

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

        # File handler
        import os
        log_file = config["app"]["log_file"]
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler = logging.FileHandler(config["app"]["log_file"], mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Cleanup on exit
        atexit.register(lambda: [h.close() for h in self.logger.handlers])

    def info(self, message):
        self.logger.info(message)

    def info_console(self, message):
        # Create temporary logger for console only
        temp_logger = logging.getLogger("console_only")
        if not temp_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            temp_logger.addHandler(handler)
            temp_logger.setLevel(logging.INFO)
        temp_logger.info(message)

    def error(self, message):
        self.logger.error(message)


custom_logger = CustomLogger()

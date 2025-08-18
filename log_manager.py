import logging
import os

class AppLogger:
    def __init__(self, log_file='debug.log', log_level=logging.DEBUG):
        #  1. Clear the log file if it exists
        if os.path.exists(log_file):
            open(log_file, 'w').close()

        #  2. Remove all existing handlers to prevent duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        #  3. Create log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        #  4. File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        #  5. Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        #  6. Set up the root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


log_instance = AppLogger(log_file='debug_matmul.log')

#######################
# Unit Testing
#######################
if __name__ == "__main__":
    arr = [1, 2, 3, 4]

    # initiating the log obj
    log = AppLogger(log_file='test_debug.log')

    # Actual Logging 
    log.info("#" * 20)
    log.info(f"arr : {arr}")
    log.info("#" * 20)


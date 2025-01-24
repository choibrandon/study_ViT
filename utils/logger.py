import os
import logging

class Logger:
    def __init__(self, log_dir='./logs',log_file='log.txt'):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)

if __name__ == '__main__':
    def main():
        logger = Logger()
        logger.log('This is a test log message.')
    main()
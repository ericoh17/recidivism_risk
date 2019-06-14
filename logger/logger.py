import logging
import os
import time

def create_logger(save_file):
  
  # initialize logger
  logger = logging.getLogger("recidivism")
  logger.setLevel(logging.INFO)

  # create the logging file handler
  file_handler = logging.FileHandler(save_file)

  # create the logging console handler
  stream_handler = logging.StreamHandler()

  # formatting
  formatter = logging.Formatter("%(asctime)s - %(message)s")
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)

  # add handlers to logger object
  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)

  return logger

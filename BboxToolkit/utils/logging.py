# Modified from mmcv.utils.logging
# Original project: https://github.com/open-mmlab/mmcv
import logging
from cloghandler import ConcurrentRotatingFileHandler

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., loger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a"
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = ConcurrentRotatingFileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s -%(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        logger = get_logger(logger)
        logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but get{type(logger)}')

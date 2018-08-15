import logging
import os

from utils.misc import get_nice_logger


class TestGetNiceLogger:
    def test_get_nice_logger_standard(self):
        log_filepath = '/tmp/test_nice_logger.log'
        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        logger = get_nice_logger('name', log_filepath=log_filepath)
        logger.info('first test message')
        logger.debug('debug message')
        logger.error('error message')

        with open(log_filepath, 'r') as f:
            logged_lines = [line for line in f]

        timestamp_length = len('0000-00-00 00:00:00')

        assert len(logged_lines) == 3
        assert logged_lines[0][timestamp_length + 1:].startswith('INFO')
        assert logged_lines[1][timestamp_length + 1:].startswith('DEBUG')
        assert logged_lines[2][timestamp_length + 1:].startswith('ERROR')

    def test_get_nice_logger_append(self):
        """ Will the logger append to a file correctly?"""
        log_filepath = '/tmp/test_nice_logger.log'
        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        logger = get_nice_logger('name', log_filepath=log_filepath)
        logger.info('a')
        logger.info('b')

        # Remove logger
        logger.handlers = []
        del logger

        logger = get_nice_logger('name', log_filepath=log_filepath,
                                 file_exists_behavior='append')
        logger.info('c')
        logger.info('d')

        with open(log_filepath, 'r') as f:
            logged_lines = [line for line in f]

        assert len(logged_lines) == 4

    def test_get_nice_logger_turn_off_debug_mode(self):
        log_filepath = '/tmp/test_nice_logger.log'
        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        logger = get_nice_logger('name', log_filepath=log_filepath)
        logger.setLevel(logging.INFO)

        logger.info('first test message')
        logger.debug('debug message')
        logger.error('error message')

        with open(log_filepath, 'r') as f:
            logged_lines = [line for line in f]

        timestamp_length = len('0000-00-00 00:00:00')

        assert len(logged_lines) == 2
        assert logged_lines[0][timestamp_length + 1:].startswith('INFO')
        assert logged_lines[1][timestamp_length + 1:].startswith('ERROR')

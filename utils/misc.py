import logging
import pickle
import re
import sys


def get_nice_logger(name, log_filepath, file_exists_behavior='append'):
    """
    Returns a Logger which writes to a file but also to system out.

    :param name: Name of the logger
    :param log_filepath: Path to the file where the logs should be written
    :param file_exists_behavior: One of 'overwrite' and 'append' - controls
                                 whether to replace the logging file.
    :return: The Logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if file_exists_behavior == 'overwrite':
        file_mode = 'w'
    elif file_exists_behavior == 'append':
        file_mode = 'a'
    else:
        raise ValueError('file_mode_behavior has to be one of "overwrite" or "append".')

    logging_file_handler = logging.FileHandler(log_filepath, mode=file_mode)
    logging_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(logging_file_handler)
    logging_stdout_handler = logging.StreamHandler(sys.stdout)
    logging_stdout_handler.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s :: %(message)s",
        "%Y-%m-%d %H:%M:%S")
    logging_file_handler.setFormatter(log_formatter)
    logging_stdout_handler.setFormatter(log_formatter)
    logger.addHandler(logging_stdout_handler)
    logger.addHandler(logging_file_handler)
    return logger


class ObjectSaver:
    def __init__(self, path):
        self.path = path

        try:
            with open(path, 'rb') as f:
                self.objects = pickle.load(f)
            if type(self.objects) is dict:
                print('Restored objects: {} from path {}'.format(
                    list(self.objects.keys()),
                    self.path
                ))
            else:
                print('Restored {} object from path {}'.format(
                    type(self.objects),
                    self.path
                ))

        except FileNotFoundError:
            # make new dictionary
            self.objects = dict()

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.objects, f)
        print('Saved pickled objects at {}'.format(self.path))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Taken from https://stackoverflow.com/questions/5967500/how-to-correctly-
               sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def flatten_list_one_level(xs):
    return [x for ys in xs for x in ys]

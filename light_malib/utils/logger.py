# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang
# Copyright (c) 2021 MARL @ SJTU

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
from colorlog import ColoredFormatter
from logging import LogRecord, Logger

LOG_LEVEL = logging.INFO


class MyLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        super().__init__(logger, extra)

    def process(self, msg, kwargs):

        if "tag" in self.extra:
            msg = "{}\ntags: {}".format(msg, self.extra["tags"])

        return msg, kwargs


class LoggerFactory:
    @staticmethod
    def build_logger(name="Light-MALib"):
        Logger = logging.getLogger(name)

        Logger.setLevel(LOG_LEVEL)
        Logger.handlers = []  # No duplicated handlers
        Logger.propagate = False  # workaround for duplicated logs in ipython

        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white,bold",
                "INFOV": "cyan,bold",
                "WARNING": "yellow",
                "ERROR": "red,bold",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOG_LEVEL)
        stream_handler.setFormatter(formatter)
        Logger.addHandler(stream_handler)
        return Logger

    @staticmethod
    def add_file_handler(Logger, filepath):
        file_handler = logging.FileHandler(filepath, mode="a")

        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s <pid: %(process)d, tid: %(thread)d, module: %(module)s, func: %(funcName)s>",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white,bold",
                "INFOV": "cyan,bold",
                "WARNING": "yellow",
                "ERROR": "red,bold",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        Logger.addHandler(file_handler)
        return Logger

    @staticmethod
    def get_logger(name="Light-MALib", extra=None):
        logger = logging.getLogger(name)
        if extra is not None:
            logger = MyLoggerAdapter(logger, extra)
        return logger


Logger = LoggerFactory.build_logger()

# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

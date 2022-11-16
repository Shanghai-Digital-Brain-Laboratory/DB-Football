# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

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

from collections import defaultdict
import pprint
from unicodedata import category
from light_malib.utils.logger import Logger
from functools import wraps


class Registry:
    TRAINER = "Trainer"
    LOSS = "Loss"
    POLICY = "Policy"
    ENV = "Env"
    STOPPER = "Stopper"

    def __init__(self):
        self.registries = defaultdict(dict)
        self.loaded = False

    def load(self):
        if not self.loaded:
            from . import registration
        self.loaded = True

    def registered(self, category, name=None):
        def wrapper(wrapperd_cls):
            registered_name = wrapperd_cls.__name__ if name is None else name
            self.register(category, registered_name, wrapperd_cls)
            return wrapperd_cls

        return wrapper

    def pprint(self, category=None):
        self.load()
        if category is not None:
            pprint.pprint(self.registries[category])
        else:
            pprint.pprint(self.registries)

    def register(self, category, name, data):
        if name in self.registries[category]:
            Logger.error(
                "{} is already registered in category {}!".format(name, category)
            )
            return
        self.registries[category][name] = data

    def get(self, category, name):
        self.load()
        if category not in self.registries:
            Logger.error("category {} is not found!".format(category))
            return
        if name not in self.registries[category]:
            Logger.error("{} is not registered in category {}!".format(name, category))
        return self.registries[category][name]


registry = Registry()

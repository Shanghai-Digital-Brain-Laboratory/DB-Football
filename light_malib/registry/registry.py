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

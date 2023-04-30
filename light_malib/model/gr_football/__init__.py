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

from . import built_in
from . import basic
from . import basic_enhanced
from . import enhanced_LightActionMask_5
from ._legacy import enhanced_extended, enhanced_LightActionMask_11
from ._legacy.enhanced_LightActionMask_11 import PartialLayernorm
import sys
import functools

# NOTE(jh): refer to https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
def partial_class(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return NewCls

# TODO(jh): the following implementation may be temporary and for backward compatibility.
class ModelWrapper:
    def __init__(self,Actor,Critic,FeatureEncoder, **kwargs) -> None:
        self.Actor=Actor
        self.Critic=Critic
        self.FeatureEncoder=FeatureEncoder
        for attr_name, obj in kwargs.items():
            setattr(self, attr_name, obj)

        self.__name__ = 'model_wrapper'

sys.modules["light_malib.model.gr_football.built_in_5"]=built_in
sys.modules["light_malib.model.gr_football.built_in_11"]=built_in
sys.modules["light_malib.model.gr_football.basic_5"]=ModelWrapper(
    basic.Actor,
    basic.Critic,
    partial_class(basic.FeatureEncoder,num_players=5*2)
)

sys.modules["light_malib.model.gr_football.enhanced_LightActionMask_5"]=ModelWrapper(
    enhanced_LightActionMask_5.Actor,
    enhanced_LightActionMask_5.Critic,
    enhanced_LightActionMask_5.FeatureEncoder
)



sys.modules["light_malib.model.gr_football.basic_11"]=ModelWrapper(
    basic.Actor,
    basic.Critic,
    partial_class(basic.FeatureEncoder,num_players=11*2)
)
sys.modules["light_malib.model.gr_football.basic_enhanced_11"]=ModelWrapper(
    basic_enhanced.Actor,
    basic_enhanced.Critic,
    partial_class(basic.FeatureEncoder,num_players=11*2)
)
sys.modules["light_malib.model.gr_football.enhanced_LightActionMask_11"]=ModelWrapper(
    enhanced_LightActionMask_11.Actor,
    enhanced_LightActionMask_11.Critic,
    enhanced_LightActionMask_11.FeatureEncoder,
    PartialLayernorm = PartialLayernorm
)
sys.modules["light_malib.model.gr_football.enhanced_extended"]=ModelWrapper(
    enhanced_extended.Actor,
    enhanced_extended.Critic,
    enhanced_extended.FeatureEncoder,
    PartialLayernorm=PartialLayernorm
)
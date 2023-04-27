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


from torch import nn
import torch

from light_malib.utils.typing import List
from light_malib.utils.preprocessor import get_preprocessor

from ..common.model import get_model
from ..utils import RNNLayer, init_fc_weights


class RNNNet(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        self.base = get_model(model_config)(
            observation_space,
            None,
            custom_config.get("use_cuda", False),
            use_feature_normalization=custom_config["use_feature_normalization"],
        )
        fc_last_hidden = model_config["layers"][-1]["units"]
        self.feat_dim = fc_last_hidden

        act_dim = get_preprocessor(action_space)(action_space).size
        self.act_dim = act_dim
        self.out = nn.Linear(fc_last_hidden, act_dim)

        use_orthogonal = initialization["use_orthogonal"]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_weights(m):
            if type(m) == nn.Linear:
                init_fc_weights(m, init_method, initialization["gain"])

        self.base.apply(init_weights)
        self.out.apply(init_weights)
        self._use_rnn = custom_config["use_rnn"]
        if self._use_rnn:
            self.rnn = RNNLayer(
                fc_last_hidden,
                fc_last_hidden,
                custom_config["rnn_layer_num"],
                use_orthogonal,
            )
        self.rnn_state_size = fc_last_hidden
        self.rnn_layer_num = custom_config["rnn_layer_num"]

    def forward(self, obs, rnn_states, masks):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if rnn_states is not None:
            rnn_states = torch.as_tensor(rnn_states, dtype=torch.float32)
        feat = self.base(obs)
        if self._use_rnn:
            assert masks is not None
            masks = torch.as_tensor(masks, dtype=torch.float32)
            feat, rnn_states = self.rnn(feat, rnn_states, masks)

        act_out = self.out(feat)
        return act_out, rnn_states

    def get_initial_state(self) -> List[torch.TensorType]:
        # FIXME(ming): ...
        return [self._init_hidden()]

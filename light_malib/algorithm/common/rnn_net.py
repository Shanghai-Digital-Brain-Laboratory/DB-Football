# MIT License

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

        act_dim = act_dim = get_preprocessor(action_space)(action_space).size
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

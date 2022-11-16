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

from collections import OrderedDict

from light_malib import rollout, agent, training, agent, buffer
from light_malib.agent import AgentManager
from light_malib.agent.agent import Agent, Agents
from light_malib.evaluation.evaluation_manager import EvaluationManager
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.framework.scheduler.psro_scheduler import PSROScheduler
from light_malib.framework.scheduler.seq_league_scheduler import SeqLeagueScheduler
from light_malib.utils.desc.task_desc import TrainingDesc
import ray
import numpy as np
from light_malib.utils.distributed import get_resources
from light_malib.utils.logger import Logger


class PBTRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.framework_cfg = self.cfg.framework
        self.id = self.framework_cfg.name

        ###### Initialize Components #####

        RolloutManager = ray.remote(
            **get_resources(cfg.rollout_manager.distributed.resources)
        )(rollout.RolloutManager)
        TrainingManager = ray.remote(
            **get_resources(cfg.training_manager.distributed.resources)
        )(training.TrainingManager)
        DataServer = ray.remote(**get_resources(cfg.data_server.distributed.resources))(
            buffer.DataServer
        )
        PolicyServer = ray.remote(
            **get_resources(cfg.policy_server.distributed.resources)
        )(buffer.PolicyServer)

        # the order of creation is important? cannot have circle reference
        # create agents
        agents = AgentManager.build_agents(self.cfg.agent_manager)

        self.data_server = DataServer.options(
            name="DataServer", max_concurrency=500
        ).remote("DataServer", self.cfg.data_server)

        self.policy_server = PolicyServer.options(
            name="PolicyServer", max_concurrency=500
        ).remote("PolicyServer", self.cfg.policy_server, agents)

        self.rollout_manager = RolloutManager.options(
            name="RolloutManager", max_concurrency=500
        ).remote("RolloutManager", self.cfg.rollout_manager, agents)

        self.training_manager = TrainingManager.options(
            name="TrainingManager", max_concurrency=50
        ).remote("TrainingManager", self.cfg.training_manager)

        # NOTE: self.agents is not shared with remote actors.
        self.agent_manager = AgentManager(self.cfg.agent_manager)
        self.policy_data_manager = PolicyDataManager(
            self.cfg.policy_data_manager, self.agent_manager
        )
        self.evaluation_manager = EvaluationManager(
            self.cfg.evaluation_manager, self.agent_manager, self.policy_data_manager
        )

        # TODO(jh): scheduler is designed for future distributed purposes.
        if self.id == "psro":
            self.scheduler = PSROScheduler(
                self.cfg.framework, self.agent_manager, self.policy_data_manager
            )
        elif self.id == "seq_league":
            self.scheduler = SeqLeagueScheduler(
                self.cfg.framework, self.agent_manager, self.policy_data_manager
            )
        else:
            raise NotImplementedError

        Logger.info("PBTRunner {} initialized".format(self.id))

    def run(self):
        self.scheduler.initialize(self.cfg.populations)
        if self.cfg.eval_only:
            self.evaluation_manager.eval()
        else:
            while True:
                self.evaluation_manager.eval()
                training_desc = self.scheduler.get_task()
                if training_desc is None:
                    break
                Logger.info("training_desc: {}".format(training_desc))
                training_task_ref = self.training_manager.train.remote(training_desc)
                ray.get(training_task_ref)
                self.scheduler.submit_result(None)
            Logger.info("PBTRunner {} ended".format(self.id))

    def close(self):
        ray.get(self.training_manager.close.remote())
        ray.get(self.rollout_manager.close.remote())

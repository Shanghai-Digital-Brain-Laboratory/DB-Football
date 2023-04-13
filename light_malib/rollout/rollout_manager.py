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
import os
import threading
from typing import List
import numpy as np
from light_malib.utils.desc.policy_desc import PolicyDesc
from light_malib.utils.logger import Logger
from light_malib.utils.distributed import get_actor, get_resources
from light_malib.agent.agent import Agents
from . import rollout_worker
import ray
import queue
from light_malib.utils.desc.task_desc import RolloutDesc, RolloutEvalDesc
from light_malib.utils.decorator import limited_calls
import traceback
from light_malib.utils.timer import global_timer
from light_malib.utils.metric import Metrics


class RolloutManager:
    def __init__(self, id, cfg, agents: Agents):
        self.id = id
        self.cfg = cfg
        self.agents = agents

        self.policy_server = get_actor(self.id, "PolicyServer")
        self.data_server = get_actor(self.id, "DataServer")
        self.monitor = get_actor(self.id, "Monitor")
        self.traning_manager = get_actor(self.id, "TrainingManager")

        RolloutWorker = ray.remote(**get_resources(cfg.worker.distributed.resources))(
            rollout_worker.RolloutWorker
        )
        self.rollout_workers = [
            RolloutWorker.remote(
                self.default_rollout_worker_id(id),
                (self.cfg.seed * 13 + id * 1000),
                self.cfg.worker,
                self.agents,
            )
            for id in range(self.cfg.num_workers)
        ]

        self.worker_pool = ray.util.ActorPool(self.rollout_workers)

        self.stop_flag = True
        self.stop_flag_lock = threading.Lock()
        # cannot start two rollout tasks
        self.semaphore = threading.Semaphore(value=1)

        self.batch_size = self.cfg.batch_size
        self.data_buffer_max_size = self.batch_size * 5

        self.eval_batch_size = self.cfg.eval_batch_size
        self.eval_data_buffer_max_size = self.eval_batch_size * 2

        self.min_samples = self.cfg.min_samples

        self.eval_freq = self.cfg.eval_freq

        self.rollout_epoch = 0
        self.rollout_epoch_lock = threading.Lock()
        Logger.info("{} initialized".format(self.id))

    @staticmethod
    def default_rollout_worker_id(id):
        return "rollout_worker_{}".format(id)

    def sync_epoch(self):
        self.sync_epoch_event.set()

    def rollout_batch(self, batch_size, rollout_desc: RolloutDesc, eval, rollout_epoch):
        rollout_descs = [
            RolloutDesc(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.policy_distributions,
                rollout_desc.share_policies,
                sync=False,
                stopper=None,
            )
        ] * batch_size
        rollout_results = self.worker_pool.map_unordered(
            lambda worker, rollout_desc: worker.rollout.remote(
                rollout_desc, eval, rollout_epoch
            ),
            values=rollout_descs,
        )
        return rollout_results

    def _rollout_loop(self, rollout_desc: RolloutDesc):
        with self.stop_flag_lock:
            assert self.stop_flag
            self.stop_flag = False

        with self.rollout_epoch_lock:
            rollout_epoch = self.rollout_epoch

        if rollout_desc.sync:
            return

        # TODO(jh): currently async rollout doesn't support evaluation
        submit_ctr = 0
        for _ in range(self.cfg.num_workers):
            self.worker_pool.submit(
                lambda worker, v: worker.rollout.remote(rollout_desc, eval = False,
                                                        rollout_epoch=rollout_epoch),
                value=None,
            )
            submit_ctr += 1

        while True:
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            # wait for a rollout to be complete
            result = self.worker_pool.get_next_unordered()
            # start a new task for this available process
            with self.rollout_epoch_lock:
                rollout_epoch = self.rollout_epoch
            self.worker_pool.submit(
                lambda worker, v: worker.rollout.remote(
                    rollout_desc, eval=False, rollout_epoch=rollout_epoch
                ),
                value=None,
            )
            submit_ctr += 1
            with self.data_buffer_lock:
                self.data_buffer.put_nowait(result)
                while self.data_buffer.qsize() > self.data_buffer_max_size:
                    self.data_buffer.get_nowait()
                if self.data_buffer.qsize() >= self.batch_size:
                    self.data_buffer_ready.notify()

        # FIXME(jh) we have to wait all tasks to terminate? any better way?
        while True:
            if self.worker_pool.has_next():
                self.worker_pool.get_next_unordered()
            else:
                break

        self.data_buffer = None
        self.data_buffer_lock = None
        self.data_buffer_ready = None

    def stop_rollout(self):
        with self.stop_flag_lock:
            self.stop_flag = True

    @limited_calls("semaphore")
    def rollout(self, rollout_desc: RolloutDesc):
        self.data_buffer = queue.Queue()
        self.data_buffer_lock = threading.Lock()
        self.data_buffer_ready = threading.Condition(self.data_buffer_lock)
        if rollout_desc.sync:
            self.eval_data_buffer = queue.Queue()
            self.eval_data_buffer_lock = threading.Lock()
            self.eval_data_buffer_ready = threading.Condition(
                self.eval_data_buffer_lock
            )
            self.sync_epoch_event = threading.Event()

        with self.rollout_epoch_lock:
            self.rollout_epoch = 0

        self.expr_log_dir = ray.get(self.monitor.get_expr_log_dir.remote())
        self.agent_id = rollout_desc.agent_id
        self.policy_id = rollout_desc.policy_id

        # only used for async
        self._rollout_loop_thread = threading.Thread(
            target=self._rollout_loop, args=(rollout_desc,)
        )
        self._rollout_loop_thread.start()

        stopper = rollout_desc.stopper

        # TODO use stopper
        try:
            best_reward = -np.inf
            self.rollout_metrics = Metrics(self.cfg.rollout_metric_cfgs)
            while True:
                # TODO(jh): ...
                stopper_kwargs = {"step": self.rollout_epoch}
                stopper_kwargs.update(self.rollout_metrics.get_means())
                with self.rollout_epoch_lock:
                    if stopper.stop(**stopper_kwargs):
                        break
                    self.rollout_epoch += 1
                    rollout_epoch = self.rollout_epoch

                Logger.info("Rollout {}".format(rollout_epoch))

                global_timer.record("batch_start")
                if rollout_desc.sync:
                    self.sync_epoch_event.clear()
                    rollout_results = self.rollout_batch(
                        self.batch_size,
                        rollout_desc,
                        eval=False,
                        rollout_epoch=rollout_epoch,
                    )
                    self.put_batch(
                        self.data_buffer,
                        self.data_buffer_lock,
                        self.data_buffer_ready,
                        self.batch_size,
                        self.data_buffer_max_size,
                        rollout_results,
                    )
                results, timer_results = self.get_batch(
                    self.data_buffer,
                    self.data_buffer_lock,
                    self.data_buffer_ready,
                    self.batch_size,
                )
                global_timer.time("batch_start", "batch_end", "batch")
                timer_results.update(global_timer.elapses)
                global_timer.clear()

                # log to tensorboard, etc...
                main_tag = "Rollout/{}/{}/".format(
                    rollout_desc.agent_id, rollout_desc.policy_id
                )
                ray.get(
                    self.monitor.add_multiple_scalars.remote(
                        main_tag, results, global_step=rollout_epoch
                    )
                )
                main_tag = "RolloutTimer/{}/{}/".format(
                    rollout_desc.agent_id, rollout_desc.policy_id
                )
                ray.get(
                    self.monitor.add_multiple_scalars.remote(
                        main_tag, timer_results, global_step=rollout_epoch
                    )
                )

                # save model periodically
                if rollout_epoch % self.cfg.saving_interval == 0:
                    self.save_current_model(f"epoch_{rollout_epoch}")

                if rollout_desc.sync and rollout_epoch % self.eval_freq == 0:
                    Logger.info(
                        "Rollout Eval {} eval {} rollouts".format(
                            rollout_epoch, self.eval_batch_size
                        )
                    )

                    rollout_results = self.rollout_batch(
                        self.eval_batch_size,
                        rollout_desc,
                        eval=True,
                        rollout_epoch=rollout_epoch,
                    )
                    self.put_batch(
                        self.eval_data_buffer,
                        self.eval_data_buffer_lock,
                        self.eval_data_buffer_ready,
                        self.eval_batch_size,
                        self.eval_data_buffer_max_size,
                        rollout_results,
                    )
                    results, timer_results = self.get_batch(
                        self.eval_data_buffer,
                        self.eval_data_buffer_lock,
                        self.eval_data_buffer_ready,
                        self.eval_batch_size,
                    )

                    # log to tensorboard, etc...
                    main_tag = "RolloutEval/{}/{}/".format(
                        rollout_desc.agent_id, rollout_desc.policy_id
                    )
                    ray.get(
                        self.monitor.add_multiple_scalars.remote(
                            main_tag, results, global_step=rollout_epoch
                        )
                    )
                    main_tag = "RolloutEvalTimer/{}/{}/".format(
                        rollout_desc.agent_id, rollout_desc.policy_id
                    )
                    ray.get(
                        self.monitor.add_multiple_scalars.remote(
                            main_tag, timer_results, global_step=rollout_epoch
                        )
                    )

                    # TODO(jh): track some important stats within a window
                    self.rollout_metrics.update(results)

                if not rollout_desc.sync:
                    # TODO(jh): currently eval is not supported in async, so we use rollout stats instead
                    self.rollout_metrics.update(results)

                # save best stable model
                rollout_metrics_mean = self.rollout_metrics.get_means(
                    metric_names=["reward", "win"]
                )
                reward = rollout_metrics_mean["reward"]
                win = rollout_metrics_mean["win"]
                if reward >= best_reward:
                    Logger.warning(
                        f"save the best model(average reward:{reward},average win:{win})"
                    )
                    best_reward = reward
                    policy_desc = self.pull_policy(self.agent_id, self.policy_id)
                    best_policy_desc = PolicyDesc(
                        self.agent_id,
                        f"{self.policy_id}.best",
                        policy_desc.policy,
                        version=rollout_epoch,
                    )
                    ray.get(self.policy_server.push.remote(self.id, best_policy_desc))

                if (
                    rollout_desc.sync
                    and rollout_epoch * self.batch_size >= self.min_samples
                ):
                    self.sync_epoch_event.wait()

        except Exception as e:
            # save model
            self.save_current_model("exception")
            Logger.error(traceback.format_exc())
            raise e

        Logger.warning(
            f"save the last model(average reward:{reward},average win:{win})"
        )
        # save the last model
        self.save_current_model("last")

        # save the best model
        best_policy_desc = self.pull_policy(self.agent_id, f"{self.policy_id}.best")
        self.save_model(best_policy_desc.policy, self.agent_id, self.policy_id, "best")
        # also push to remote to replace the last policy
        best_policy_desc.policy_id = self.policy_id
        best_policy_desc.version = float("inf")  # a version for freezing
        ray.get(self.policy_server.push.remote(self.id, best_policy_desc))

        # signal tranining_manager to stop training
        ray.get(self.traning_manager.stop_training.remote())

        # training_manager will stop rollout loop, wait here
        self._rollout_loop_thread.join()

        # softly wait for training loop ends
        training_loop_stopped = ray.get(
            self.traning_manager.training_loop_stopped.remote()
        )
        while not training_loop_stopped:
            rollout_results = self.rollout_batch(
                self.batch_size, rollout_desc, eval=False, rollout_epoch=rollout_epoch
            )
            list(rollout_results)
            training_loop_stopped = ray.get(
                self.traning_manager.training_loop_stopped.remote()
            )

        Logger.warning("Rollout ends after {} epochs".format(self.rollout_epoch))

    def pull_policy(self, agent_id, policy_id):
        if policy_id not in self.agents[agent_id].policy_data:
            policy_desc = ray.get(
                self.policy_server.pull.remote(
                    self.id, agent_id, policy_id, old_version=None
                )
            )
            self.agents[agent_id].policy_data[policy_id] = policy_desc
        else:
            old_policy_desc = self.agents[agent_id].policy_data[policy_id]
            policy_desc = ray.get(
                self.policy_server.pull.remote(
                    self.id, agent_id, policy_id, old_version=old_policy_desc.version
                )
            )
            if policy_desc is not None:
                self.agents[agent_id].policy_data[policy_id] = policy_desc
            else:
                policy_desc = old_policy_desc
        return policy_desc

    def save_current_model(self, name):
        self.pull_policy(self.agent_id, self.policy_id)
        policy_desc = self.agents[self.agent_id].policy_data[self.policy_id]
        if policy_desc is not None:
            return self.save_model(
                policy_desc.policy, self.agent_id, self.policy_id, name
            )

    def save_model(self, policy, agent_id, policy_id, name):
        dump_dir = os.path.join(self.expr_log_dir, agent_id, policy_id, name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        policy.dump(dump_dir)
        Logger.info(
            "Saving model {} {} {} to {}".format(agent_id, policy_id, name, dump_dir)
        )
        return policy

    @limited_calls("semaphore")
    def rollout_eval(self, rollout_eval_desc: RolloutEvalDesc):
        policy_combinations = rollout_eval_desc.policy_combinations
        num_eval_rollouts = rollout_eval_desc.num_eval_rollouts
        # prepare rollout_desc
        # agent_id & policy_id here is dummy
        rollout_descs = [
            RolloutDesc(
                agent_id="agent_0",
                policy_id=policy_combination["agent_0"],
                policy_distributions=policy_combination,
                share_policies=rollout_eval_desc.share_policies,
                sync=False,
                stopper=None,
            )
            for policy_combination in policy_combinations
        ]
        rollout_descs *= num_eval_rollouts

        rollout_results = self.worker_pool.map_unordered(
            lambda worker, rollout_desc: worker.rollout.remote(rollout_desc, eval=True, rollout_epoch=0),
            values=rollout_descs,
        )

        # reduce
        results = self.reduce_rollout_eval_results(rollout_results)
        return results

    def get_batch(
        self,
        data_buffer: queue.Queue,
        data_buffer_lock: threading.Lock,
        data_buffer_ready: threading.Condition,
        batch_size: int,
    ):
        # retrieve data from data buffer
        while True:
            with data_buffer_lock:
                data_buffer_ready.wait_for(lambda: data_buffer.qsize() >= batch_size)
                if data_buffer.qsize() >= batch_size:
                    rollout_results = [
                        data_buffer.get_nowait() for i in range(batch_size)
                    ]
                    break

        # reduce
        results = self.reduce_rollout_results(rollout_results)
        return results

    def put_batch(
        self,
        data_buffer: queue.Queue,
        data_buffer_lock: threading.Lock,
        data_buffer_ready: threading.Condition,
        batch_size: int,
        data_buffer_max_size: int,
        batch: List,
    ):
        with data_buffer_lock:
            for data in batch:
                data_buffer.put_nowait(data)
            while data_buffer.qsize() > data_buffer_max_size:
                data_buffer.get_nowait()
            if data_buffer.qsize() >= batch_size:
                data_buffer_ready.notify()

    def reduce_rollout_results(self, rollout_results):
        results = defaultdict(list)
        for rollout_result in rollout_results:
            # TODO(jh): policy-wise stats
            # NOTE(jh): now in training, we only care about statistics of the agent is trained
            main_agent_id = rollout_result["main_agent_id"]
            # policy_ids=rollout_result["policy_ids"]
            stats = rollout_result["stats"][main_agent_id]
            for k, v in stats.items():
                results[k].append(v)

        for k, v in results.items():
            results[k] = np.mean(v)

        timer_results = defaultdict(list)
        for rollout_result in rollout_results:
            timer = rollout_result["timer"]
            for k, v in timer.items():
                timer_results[k].append(v)

        for k, v in timer_results.items():
            timer_results[k] = np.mean(v)

        return results, timer_results

    def reduce_rollout_eval_results(self, rollout_results):
        # {policy_comb: {agent_id: key: [value]}}
        # policy_comb = ((agent_id, policy_id),)
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for rollout_result in rollout_results:
            policy_ids = rollout_result["policy_ids"]
            stats = rollout_result["stats"]
            policy_comb = tuple(
                [(agent_id, policy_id) for agent_id, policy_id in policy_ids.items()]
            )
            for agent_id, agent_stats in stats.items():
                for key, value in agent_stats.items():
                    results[policy_comb][agent_id][key].append(value)

        for policy_comb, stats in results.items():
            for agent_id, agent_stats in stats.items():
                for key, value in agent_stats.items():
                    agent_stats[key] = np.mean(value)

        return results

    def close(self):
        if not self.stop_flag:
            try:
                self.save_current_model("{}".format(self.rollout_epoch))

                # also save the best model
                best_policy_desc = self.pull_policy(
                    self.agent_id, f"{self.policy_id}.best"
                )
                self.save_model(
                    best_policy_desc.policy, self.agent_id, self.policy_id, "best"
                )
            except Exception:
                import traceback

                Logger.error("{}".format(traceback.format_exc()))

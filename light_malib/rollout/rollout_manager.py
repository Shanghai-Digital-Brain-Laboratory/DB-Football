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
from light_malib.utils.naming import default_rollout_worker_id

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
                default_rollout_worker_id(id),
                (self.cfg.seed * 13 + id * 1000),
                self.cfg.worker,
                self.agents,
            )
            for id in range(self.cfg.num_workers)
        ]

        self.worker_pool = ray.util.ActorPool(self.rollout_workers)

        # cannot start two rollout tasks
        self.semaphore = threading.Semaphore(value=1)

        self.batch_size = self.cfg.batch_size
        self.data_buffer_max_size = self.batch_size * 5

        self.eval_batch_size = self.cfg.eval_batch_size
        self.eval_data_buffer_max_size = self.eval_batch_size * 2

        self.eval_freq = self.cfg.eval_freq

        self.rollout_epoch = 0
        self.rollout_epoch_lock = threading.Lock()
        Logger.info("{} initialized".format(self.id))
        
        self.stop_flag = True
        self.stop_flag_lock = threading.Lock()
                    
    def rollout(self, rollout_desc: RolloutDesc):
        
        self.data_buffer = queue.Queue()
        self.data_buffer_lock = threading.Lock()
        self.data_buffer_ready = threading.Condition(self.data_buffer_lock)
        self.eval_data_buffer = queue.Queue()
        self.eval_data_buffer_lock = threading.Lock()
        self.eval_data_buffer_ready = threading.Condition(self.eval_data_buffer_lock)
        
        self.rollout_desc=rollout_desc
        self.expr_log_dir = ray.get(self.monitor.get_expr_log_dir.remote())
        self.agent_id = rollout_desc.agent_id
        self.policy_id = rollout_desc.policy_id
        self.stop_flag = False
        self.stop_flag_lock = threading.Lock()
        
        if rollout_desc.sync:
            self.sync_rollout(rollout_desc)
        else:
            self.async_rollout(rollout_desc)
        
        self.data_buffer = None
        self.data_buffer_lock = None
        self.data_buffer_ready = None
        self.eval_data_buffer = None
        self.eval_data_buffer_lock = None
        self.eval_data_buffer_ready = None  
        
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
            lambda worker, rollout_desc: worker.rollout.remote(rollout_desc, eval=True),
            values=rollout_descs,
        )

        # reduce
        results = self.reduce_rollout_eval_results(rollout_results)
        return results
        
    ##### Sync Rollout START #####        
    
    def sync_rollout(self, rollout_desc: RolloutDesc):

        stopper = rollout_desc.stopper
        # TODO use stopper
        try:
            with self.rollout_epoch_lock:
                self.rollout_epoch = 0
            best_reward = -np.inf
            self.rollout_metrics = Metrics(self.cfg.rollout_metric_cfgs)
            while True:
                stopper_kwargs = {"step": self.rollout_epoch}
                stopper_kwargs.update(self.rollout_metrics.get_means())
                with self.rollout_epoch_lock:
                    if stopper.stop(**stopper_kwargs):
                        break
                    self.rollout_epoch += 1
                    rollout_epoch = self.rollout_epoch

                Logger.info("Rollout {}, Global Step {}".format(rollout_epoch, self.get_global_step(rollout_epoch)))

                global_timer.record("batch_start")
                rollout_results = self.rollout_batch(
                    self.batch_size,
                    rollout_desc,
                    eval=False,
                    rollout_epoch=rollout_epoch,
                )
                results, timer_results = self.reduce_rollout_results(rollout_results)
                
                global_timer.time("batch_start", "batch_end", "batch")
                timer_results.update(global_timer.elapses)
                global_timer.clear()
                
                self.log_to_tensorboard(results,timer_results,rollout_epoch=rollout_epoch,main_tag="Rollout")

                # save model periodically
                if rollout_epoch % self.cfg.saving_interval == 0:
                    self.save_current_model(f"epoch_{rollout_epoch}")

                if rollout_epoch % self.eval_freq == 0:
                    Logger.info(
                        "Rollout Eval {}, Global Step {}".format(
                            rollout_epoch, self.get_global_step(rollout_epoch)
                        )
                    )

                    rollout_results = self.rollout_batch(
                        self.eval_batch_size,
                        rollout_desc,
                        eval=True,
                        rollout_epoch=rollout_epoch,
                    )
                    
                    results, timer_results = self.reduce_rollout_results(rollout_results)
                    Logger.info(
                        "Rollout Eval {}: average reward: {}, average win: {}".format(
                            rollout_epoch, results["reward"], results["win"] 
                        )
                    )
                    self.log_to_tensorboard(results,timer_results,rollout_epoch=rollout_epoch,main_tag="RolloutEval")
                
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
                    self.push_best_model_to_policy_server(rollout_epoch)

                # training step: update the model    
                ray.get(self.traning_manager.train_step.remote())
                
        except Exception as e:
            # save model
            self.save_current_model("{}.exception".format(rollout_epoch))
            self.save_best_model_from_policy_server()
            Logger.error(traceback.format_exc())
            raise e

        Logger.warning(
            f"save the last model(average reward:{reward},average win:{win})"
        )
        # save the last model
        self.save_current_model("{}.last".format(rollout_epoch))
        
        self.stop_rollout()

        # save the best model
        best_policy_desc = self.save_best_model_from_policy_server()
        # also push to remote to replace the last policy
        best_policy_desc.policy_id = self.policy_id
        best_policy_desc.version = float("inf")  # a version for freezing
        ray.get(self.policy_server.push.remote(self.id, best_policy_desc))

        # signal tranining_manager to stop training
        ray.get(self.traning_manager.stop_training.remote())
        # call the last step to clean up things
        ray.get(self.traning_manager.train_step.remote())

        Logger.warning("Rollout ends after {} epochs".format(self.rollout_epoch))
        
    def rollout_batch(self, batch_size, rollout_desc: RolloutDesc, eval, rollout_epoch):
        rollout_descs = [
            RolloutDesc(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.policy_distributions,
                rollout_desc.share_policies,
                sync=rollout_desc.sync,
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

    ##### Sync Rollout END #####    
    
    ##### Async Rollout START #####
    
    def async_rollout(self, rollout_desc: RolloutDesc):

        # only used for async
        _async_rollout_loop_thread = threading.Thread(
            target=self._async_rollout_loop, args=(rollout_desc,)
        )
        _async_rollout_loop_thread.start()

        _async_training_loop = self.traning_manager.async_training_loop.remote()

        stopper = rollout_desc.stopper

        # TODO use stopper
        try:
            with self.rollout_epoch_lock:
                self.rollout_epoch = 0
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

                Logger.info("Rollout {}, Global Step {}".format(rollout_epoch,self.get_global_step(rollout_epoch)))

                global_timer.record("batch_start")
                results, timer_results = self.get_batch(
                    self.data_buffer,
                    self.data_buffer_lock,
                    self.data_buffer_ready,
                    self.batch_size,
                )
                global_timer.time("batch_start", "batch_end", "batch")
                timer_results.update(global_timer.elapses)
                global_timer.clear()

                self.log_to_tensorboard(results,timer_results,rollout_epoch,main_tag="Rollout")

                # save model periodically
                if rollout_epoch % self.cfg.saving_interval == 0:
                    self.save_current_model(f"epoch_{rollout_epoch}")
                
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
                    self.push_best_model_to_policy_server(rollout_epoch)
                    
                self.check_error([_async_training_loop])

        except Exception as e:
            # save model
            self.save_current_model("{}.exception".format(rollout_epoch))
            self.save_best_model_from_policy_server()
            Logger.error(traceback.format_exc())
            raise e

        Logger.warning(
            f"save the last model(average reward:{reward},average win:{win})"
        )
        # save the last model
        self.save_current_model("{}.last".format(rollout_epoch))

        # save the best model
        best_policy_desc = self.save_best_model_from_policy_server()
        # also push to remote to replace the last policy
        best_policy_desc.policy_id = self.policy_id
        best_policy_desc.version = float("inf")  # a version for freezing
        ray.get(self.policy_server.push.remote(self.id, best_policy_desc))

        # signal tranining_manager to stop training
        ray.get(self.traning_manager.stop_training.remote())
        # we have to wait for training loop to stop first in case that it may still need data.
        ray.get(_async_training_loop)
        
        # stop rollout loop
        self.stop_rollout()
        _async_rollout_loop_thread.join()

        Logger.warning("Rollout ends after {} epochs".format(self.rollout_epoch))
        
    def _async_rollout_loop(self, rollout_desc: RolloutDesc):
        '''
        TODO(jh): just merge this loop into the main async call.
        '''
        with self.rollout_epoch_lock:
            rollout_epoch = self.rollout_epoch

        # TODO(jh): currently async rollout doesn't support evaluation
        submit_ctr = 0
        recieve_ctr = 0
        eval_submit_ctr=0   
        eval=False

        for _ in range(self.cfg.num_workers):
            self.worker_pool.submit(
                lambda worker, v: worker.rollout.remote(rollout_desc, eval=eval, rollout_epoch=rollout_epoch),
                value=None,
            )
            submit_ctr += 1

        while True:
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            
            # wait for a rollout to be complete
            result = self.worker_pool.get_next_unordered()
            recieve_ctr += 1              

            # start a new task for this available process
            with self.rollout_epoch_lock:
                rollout_epoch = self.rollout_epoch  

            # if submit_ctr%(self.eval_freq*self.batch_size)==0:
            #     if eval_submit_ctr==0:
            #         eval_rollout_epoch=rollout_epoch
            #         Logger.info(
            #             "Rollout Eval {}, Global Step {}".format(
            #                 eval_rollout_epoch, self.get_global_step(eval_rollout_epoch)
            #             )
            #         )
            #         eval=True   
            #     elif eval_submit_ctr==self.eval_batch_size:
            #         eval_submit_ctr=0
            #         eval=False          
                
            eval=False    
                
            self.worker_pool.submit(
                lambda worker, v: worker.rollout.remote(
                    rollout_desc, eval=eval, rollout_epoch=rollout_epoch
                ),
                value=None,
            )
            
            submit_ctr+=1
            # if eval:
            #     eval_submit_ctr += 1
            # else:
            #     submit_ctr += 1
            
            # if result["eval"]:                
                # with self.eval_data_buffer_lock:
                #     self.eval_data_buffer.put_nowait(result)
                #     while self.eval_data_buffer.qsize() > self.eval_data_buffer_max_size:
                #         self.eval_data_buffer.get_nowait()
                #     if self.eval_data_buffer.qsize() >= self.eval_batch_size:
                #         rollout_results = [
                #             self.eval_data_buffer.get_nowait() for i in range(self.eval_batch_size)
                #         ]
                #         results, timer_results =self.reduce_rollout_results(rollout_results)
                #         Logger.info(
                #             "Rollout Eval {}: average reward: {}, average win: {}".format(
                #                 eval_rollout_epoch, results["reward"], results["win"] 
                #             )
                #         )
                #         self.log_to_tensorboard(results,timer_results,rollout_epoch=eval_rollout_epoch,main_tag="RolloutEval")       
            # else:
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
            
    def check_error(self, task_refs):
        ready_refs, _ = ray.wait(task_refs,timeout=0)
        if len(ready_refs) > 0:
            ray.get(ready_refs)

    def stop_rollout(self):
        with self.stop_flag_lock:
            self.stop_flag = True
            
    ##### Async Rollout END #####
    
    def get_global_step(self,rollout_epoch):
        global_step=rollout_epoch*self.batch_size*self.cfg.worker.rollout_length
        return global_step
    
    def log_to_tensorboard(self, results, timer_results, rollout_epoch, main_tag="Rollout"):
        # log to tensorboard, etc...
        global_step=self.get_global_step(rollout_epoch)
        tag = "{}/{}/{}/".format(
            main_tag, self.rollout_desc.agent_id, self.rollout_desc.policy_id
        )
        ray.get(
            self.monitor.add_multiple_scalars.remote(
                tag, results, global_step=global_step
            )
        )
        tag = "{}Timer/{}/{}/".format(
            main_tag, self.rollout_desc.agent_id, self.rollout_desc.policy_id
        )
        ray.get(
            self.monitor.add_multiple_scalars.remote(
                tag, timer_results, global_step=global_step
            )
        )  

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
    
    def push_best_model_to_policy_server(self,rollout_epoch):
        policy_desc = self.pull_policy(self.agent_id, self.policy_id)
        best_policy_desc = PolicyDesc(
            self.agent_id,
            f"{self.policy_id}.best",
            policy_desc.policy,
            version=rollout_epoch,
        )
        ray.get(self.policy_server.push.remote(self.id, best_policy_desc))
        
    def save_best_model_from_policy_server(self):
        best_policy_desc = self.pull_policy(self.agent_id, f"{self.policy_id}.best")
        self.save_model(best_policy_desc.policy, self.agent_id, self.policy_id, "best")
        return best_policy_desc
        
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
        results, timer_results  = self.reduce_rollout_results(rollout_results)
        return results, timer_results

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
            for _result in rollout_result["results"]:
                # TODO(jh): policy-wise stats
                # NOTE(jh): now in training, we only care about statistics of the agent is trained
                main_agent_id = _result["main_agent_id"]
                # policy_ids=rollout_result["policy_ids"]
                stats = _result["stats"][main_agent_id]
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
            for _result in rollout_result["results"]:
                policy_ids = _result["policy_ids"]
                stats = _result["stats"]
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
                self.save_current_model("{}.exception".format(self.rollout_epoch))
                # also save the best model
                self.save_best_model_from_policy_server()
            except Exception:
                import traceback

                Logger.error("{}".format(traceback.format_exc()))

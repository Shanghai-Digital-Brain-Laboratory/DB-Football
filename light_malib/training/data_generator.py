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

import torch
import numpy as np
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer


def simple_data_generator(data, num_mini_batch, device, shuffle=True):
    assert len(data[EpisodeKey.CUR_OBS].shape) == 4, "{}".format(
        {k: v.shape for k, v in data.items()}
    )
    len_traj, n_rollout_threads, n_agent, _ = data[EpisodeKey.CUR_OBS].shape
    batch_size = len_traj * n_rollout_threads  # * n_agent

    batch = {}
    for k in data:
        try:
            global_timer.record("data_copy_start")
            if isinstance(data[k], np.ndarray):
                batch[k] = torch.from_numpy(data[k]).to(device)
            else:
                batch[k] = data[k]
            global_timer.time("data_copy_start", "data_copy_end", "data_copy")
            batch[k] = batch[k].reshape(batch_size, *data[k].shape[2:])
        except Exception:
            Logger.error("k: {}".format(k))
            raise Exception
    # jh: special optimization
    if num_mini_batch == 1:
        for k in batch:
            # batch_size,n_agent,...
            # -> batch_size*n_agent,...
            batch[k] = batch[k].reshape(-1, *batch[k].shape[2:])
        batch = {k: v for k, v in batch.items()}
        yield batch
        return

    mini_batch_size = int(np.ceil(batch_size // num_mini_batch))
    assert mini_batch_size > 0

    if shuffle:
        rand = torch.randperm(batch_size)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
    else:
        sampler = [
            slice(i * mini_batch_size, (i + 1) * mini_batch_size)
            for i in range(num_mini_batch)
        ]
    for indices in sampler:
        tmp_batch = {}
        for k in batch:
            # batch_size,n_agent,...
            # -> batch_size*n_agent,...
            tmp_batch[k] = batch[k][indices]
            tmp_batch[k] = tmp_batch[k].reshape(-1, *tmp_batch[k].shape[2:])
        yield {k: v for k, v in tmp_batch.items()}

def simple_team_data_generator(data, num_mini_batch, device, shuffle=False):
    assert len(data[EpisodeKey.CUR_OBS].shape) == 4, "{}".format(
        {k: v.shape for k, v in data.items()}
    )
    len_traj, n_rollout_threads, n_agent, _ = data[EpisodeKey.CUR_OBS].shape
    batch_size = len_traj * n_rollout_threads  # * n_agent

    batch = {}
    for k in data:
        try:
            global_timer.record("data_copy_start")
            if isinstance(data[k], np.ndarray):
                batch[k] = torch.from_numpy(data[k]).to(device)
            else:
                batch[k] = data[k]
            global_timer.time("data_copy_start", "data_copy_end", "data_copy")
            batch[k] = batch[k].reshape(batch_size, *data[k].shape[2:])
        except Exception:
            Logger.error("k: {}".format(k))
            raise Exception
    # jh: special optimization
    if num_mini_batch == 1:
        # for k in batch:
        #     # batch_size,n_agent,...
        #     # -> batch_size*n_agent,...
        #     batch[k] = batch[k].reshape(-1, *batch[k].shape[2:])
        # global_timer.record("to_gpu_start")
        # batch = {k: v for k, v in batch.items()}
        # global_timer.time("to_gpu_start", "to_gpu_end", "to_gpu")
        yield batch
        return

    mini_batch_size = int(np.ceil(batch_size // num_mini_batch))
    assert mini_batch_size > 0

    if shuffle:
        rand = torch.randperm(batch_size)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
    else:
        sampler = [
            slice(i * mini_batch_size, (i + 1) * mini_batch_size)
            for i in range(num_mini_batch)
        ]
    for indices in sampler:
        tmp_batch = {}
        for k in batch:
            # batch_size,n_agent,...
            # -> batch_size*n_agent,...
            tmp_batch[k] = batch[k][indices]
            tmp_batch[k] = tmp_batch[k].reshape(-1, *tmp_batch[k].shape[2:])
        yield {k: v for k, v in tmp_batch.items()}

def dummy_data_generator(data, num_mini_batch, device, shuffle=False):
    # assert len(data[EpisodeKey.CUR_OBS].shape) == 4, "{}".format(
    #     {k: v.shape for k, v in data.items()}
    # )
    if len(data[EpisodeKey.CUR_OBS].shape)==4:
        batch_size, len_traj, n_agent, _ = data[EpisodeKey.CUR_OBS].shape
        select_idx = 1
    elif len(data[EpisodeKey.CUR_OBS].shape)==3:
        batch_size, n_agent, _ = data[EpisodeKey.CUR_OBS].shape
        if len(data[EpisodeKey.ACTION].shape)==2:
            data[EpisodeKey.ACTION] = data[EpisodeKey.ACTION][:,:,np.newaxis]
        select_idx = -2
    else:
        raise NotImplementedError

    if num_mini_batch == 1:

        yield data
        return

    mini_batch_size = int(np.ceil(batch_size // num_mini_batch))
    assert mini_batch_size > 0

    if shuffle:
        rand = torch.randperm(batch_size)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
    else:
        sampler = [
            slice(i * mini_batch_size, (i + 1) * mini_batch_size)
            for i in range(num_mini_batch)
        ]
    for indices in sampler:
        tmp_batch = {}
        for k in data:
            # batch_size,n_agent,...
            # -> batch_size*n_agent,...
            tmp_batch[k] = data[k][indices]
            tmp_batch[k] = tmp_batch[k].reshape(-1, *tmp_batch[k].shape[select_idx:])
        yield {k: v for k, v in tmp_batch.items()}



def recurrent_generator(data, num_mini_batch, rnn_data_chunk_length, device):
    raise NotImplementedError("jh: whoever wants to use this function needs to double check this implementation first.")
    
    batch = {k: d for k, d in data.items()}
    # batch_size,seq_length,n_agent(,n_feats*)
    assert len(batch[EpisodeKey.CUR_OBS].shape) == 4, "{}".format(
        {k: v.shape for k, v in data.items()}
    )

    batch_size, seq_length, n_agent = batch[EpisodeKey.CUR_OBS].shape[:3]
    new_seq_length = seq_length - seq_length % rnn_data_chunk_length
    data_chunks = batch_size * (new_seq_length // rnn_data_chunk_length)
    mini_batch_size = int(np.ceil(data_chunks / num_mini_batch))

    def _cast(x):
        assert (
            x.shape[0] == batch_size
            and x.shape[1] == seq_length
            and x.shape[2] == n_agent
        )
        # TODO WARNING ONCE jh: for simplicity, discard last several frames
        x = x[:, :new_seq_length, ...]
        # -> data_chunks,rnn_data_chunk_length,n_agent(,n_feats*),see below
        x = x.reshape(data_chunks, rnn_data_chunk_length, *x.shape[2:])
        return x

    for k in batch:
        if isinstance(batch[k], np.ndarray):
            batch[k] = torch.from_numpy(batch[k]).to(device)
        batch[k] = _cast(batch[k])

    # jh: special optimization
    if num_mini_batch == 1:
        for k in batch:
            if k not in ["rnn_state_0", "rnn_state_1"]:
                batch[k] = torch.transpose(batch[k], 0, 1)
                batch[k] = batch[k].reshape(-1, *batch[k].shape[3:])
            else:
                batch[k] = batch[k][:, 0, ...]
                batch[k] = batch[k].reshape(-1, *batch[k].shape[2:])
        yield {k: v for k, v in tmp_batch.items()}
        return

    rand = torch.randperm(data_chunks)
    sampler = [
        rand[i * mini_batch_size : (i + 1) * mini_batch_size]
        for i in range(num_mini_batch)
    ]
    for indices in sampler:
        tmp_batch = {}
        for k in batch:
            if k not in ["rnn_state_0", "rnn_state_1"]:
                # batch_size,rnn_data_chunk_length,n_agent,(,n_feats*)
                tmp_batch[k] = batch[k][indices]
                # rnn_data_chunk_length,batch_size,n_agent,(,n_feats*)
                tmp_batch[k] = torch.transpose(tmp_batch[k], 0, 1)
                # rnn_data_chunk_length*batch_size*n_agent,(,n_feats*)
                tmp_batch[k] = tmp_batch[k].reshape(-1, *tmp_batch[k].shape[3:])
            else:
                # batch_size,rnn_data_chunk_length,n_agent,(,n_feats*)
                # -> batch_size,n_agent,(,n_feats*) only get the first hidden
                tmp_batch[k] = batch[k][indices][:, 0, ...]
                # batch_size*n_agent,(,n_feats*)
                tmp_batch[k] = tmp_batch[k].reshape(-1, *tmp_batch[k].shape[2:])

        yield {k: v for k, v in tmp_batch.items()}

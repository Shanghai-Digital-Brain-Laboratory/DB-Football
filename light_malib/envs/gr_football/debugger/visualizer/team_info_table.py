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

import tkinter as tk
import json
import os
from functools import partial

import numpy as np
from PIL import Image, ImageTk
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter import ttk
import pandas as pd
from .translation_en import *
from ...tools import geometry


class TeamInfoTable(ttk.Treeview):
    def __init__(self, parent, team, obs, step=0) -> None:
        self.team = team
        df = self.get_df(obs, self.team, step)
        columns = list(df.columns)
        super().__init__(parent, columns=columns, show="headings", selectmode="browse")
        for i in range(len(df.columns)):
            self.column(columns[i], anchor=tk.CENTER, width=50)
            self.heading(columns[i], text=columns[i], anchor=tk.CENTER)
        self.column("action", width=50)
        self.column("role", width=50)
        self.column("dist_to_ball", width=80)
        self.column("speed/0.1s", width=100)
        self.column("coords", width=100)
        if team == "left" and "rewards" in obs:
            self.column("reward", width=100)

    def update_table(self, obs, step):
        for i in self.get_children():
            self.delete(i)
        df = self.get_df(obs, self.team, step)
        for i in range(len(df)):
            data = list(df.iloc[i])
            self.insert(
                "",
                tk.END,
                values=data,
            )

    def get_df(self, obs, team, step):
        if step == 0:
            self.left_team_roles = obs["left_team_roles"]
            self.right_team_roles = obs["right_team_roles"]
            self.n_left = obs["n_left"]
            self.n_right = obs["n_right"]
            self.n_left_control = obs["n_left_control"]
            self.n_right_control = obs["n_right_control"]
        pos = obs[f"{team}_team"]
        pos = np.array(pos).reshape(-1, 2)
        direction = obs[f"{team}_team_direction"]
        tired_factor = obs[f"{team}_team_tired_factor"]
        yellow_card = obs[f"{team}_team_yellow_card"]
        active = obs[f"{team}_team_active"]
        controls = obs["controls"][team]

        ball_pos = obs["ball"]
        dists = np.round(geometry.get_dist(pos, ball_pos[:2]), 2)

        coords = np.round(geometry.tpos(pos), 2)
        coords_speed = np.around(geometry.get_coord_speed(direction) / geometry.FPS, 2)

        speeds = np.round(geometry.get_speed(direction) / geometry.FPS, 2)
        speeds = [
            "{}({},{})".format(s, sx, sy) for s, (sx, sy) in zip(speeds, coords_speed)
        ]
        coords = ["{},{}".format(sx, sy) for (sx, sy) in coords]

        actions = []
        control_direction = []
        is_dribbling = []
        is_sprinting = []
        for i in range(len(pos)):
            if i in controls and "action" in controls[i]:
                action = controls[i]["action"]
                sticky_actions = controls[i]["sticky_actions"]
                actions.append(action)
                found = False
                for i in range(8):
                    if sticky_actions[i]:
                        control_direction.append(i)
                        found = True
                if not found:
                    control_direction.append(-1)
                is_dribbling.append(str(sticky_actions[9]))
                is_sprinting.append(str(sticky_actions[8]))
            else:
                actions.append(-1)
                control_direction.append(-1)
                is_dribbling.append("-")
                is_sprinting.append("-")

        if team == "left":
            roles = self.left_team_roles
            n = self.n_left
        else:
            roles = self.right_team_roles
            n = self.n_right

        roles = [SIMPLE_ROLES(r) + str(idx) for idx, r in enumerate(roles)]

        cards = [
            "r" if not a else "y" if y else "0" for y, a in zip(yellow_card, active)
        ]
        tired_factor = [round(f, 0) for f in tired_factor]

        # print(control_direction)
        # print(actions)

        if team == "left" and "rewards" in obs:
            rewards_ = obs["rewards"]
            indices = sorted(list(obs["controls"]["left"].keys()))
            idx2control_idx_mapping = {
                idx: control_idx for control_idx, idx in enumerate(indices)
            }
            rewards = []
            for idx in range(self.n_left):
                if idx not in obs["controls"]["left"]:
                    rewards.append("-")
                else:
                    control_idx = idx2control_idx_mapping[idx]
                    rewards.append("{:.2e}".format(rewards_[control_idx]))

        data = [
            ("tiredness", tired_factor),
            ("cards", cards),
            ("sticky_direct", map(DIRECTIONS, control_direction)),
            ("dribble", is_dribbling),
            ("sprint", is_sprinting),
            ("coords", coords),
            ("speed/0.1s", speeds),
            ("dist_to_ball", dists),
            ("role", roles),
            ("action", map(ACTIONS, actions)),
        ]
        if team == "left" and "rewards" in obs:
            data.append(("reward", rewards))
        if team == "right":
            data.reverse()
        df = pd.DataFrame(dict(data))

        return df

    def pad(self, arr, width=4):
        ret = [str(d).rjust(width) for d in arr]
        return ret

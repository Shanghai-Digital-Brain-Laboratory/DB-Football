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


class BallInfoTable(ttk.Treeview):
    def __init__(self, parent, team, obs, step=0) -> None:
        self.team = team
        df = self.get_df(obs, self.team, step)
        columns = list(df.columns)
        super().__init__(parent, columns=columns, show="headings", selectmode="browse")
        for i in range(len(df.columns)):
            self.column(columns[i], anchor=tk.CENTER, width=50)
            self.heading(columns[i], text=columns[i], anchor=tk.CENTER)
        # self.column("速度/0.1s",width=100)
        # self.column("坐标",width=100)

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

        coords = np.round(geometry.tpos(obs["ball"]), 2)
        coord_speeds = np.round(
            geometry.get_coord_speed(obs["ball_direction"]) / geometry.FPS, 2
        )
        speed = np.round(geometry.get_speed(obs["ball_direction"]) / geometry.FPS, 2)
        rotations = np.round(obs["ball_rotation"], 2)

        rotations = ["{},{},{}".format(*rotations)]

        data = [
            ("rotations", rotations),
            ("speed/0.1s", speed),
            ("speed_x/0.1s", coord_speeds[0]),
            ("speed_y/0.1s", coord_speeds[1]),
            ("speed_z/0.1s", coord_speeds[2]),
            ("coord_x", [coords[0]]),
            ("coord_y", [coords[1]]),
            ("coord_z", [coords[2]]),
        ]
        # if team=="right":
        #     data.reverse()
        df = pd.DataFrame(dict(data))

        return df

    def pad(self, arr, width=4):
        ret = [str(d).rjust(width) for d in arr]
        return ret

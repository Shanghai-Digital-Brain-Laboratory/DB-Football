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


class RawDrawer:
    def __init__(self, canvas: tk.Canvas, obs: dict) -> None:
        """
        obs at step 0
        """
        self.canvas = canvas
        self.pitch_height_coord = 0.84
        self.pitch_width_coord = 2
        self.door_min_coord = -0.044
        self.door_max_coord = 0.044
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.pitch_height_frame = 360 * 0.96  # int(self.canvas_height*0.98)
        self.pitch_width_frame = 550 * 0.96  # int(self.canvas_width*0.98)
        assert (
            self.canvas_height > self.pitch_height_frame
            and self.canvas_width > self.pitch_width_frame
        )
        self.door_d = 5
        self.x_off = (self.canvas_width - self.pitch_width_frame) / 2
        self.y_off = (self.canvas_height - self.pitch_height_frame) / 2
        self.right_player_color = "blue"
        self.left_player_color = "green"
        self.ball_color = "black"
        self.player_direction_scale = 10
        self.ball_direction_scale = 10
        self.mark_r = 10
        self.mark_color = "red"
        self.pitch_color = "black"
        self.door_color = "black"
        self.info_color = "orange"
        self.info_x_off = 0
        self.info_y_off = -100
        self.ball_largest_r = 8
        self.ball_r = 4
        self.ball_largest_z = 3

        self.buffers = []

        self.create(obs)

    def tr(self, z):
        return min(
            z / self.ball_largest_z * (self.ball_largest_r - self.ball_r) + self.ball_r,
            self.ball_largest_r,
        )

    def create_ball(self, obs):
        self.draw_ball_info(obs)
        ball_x, ball_y, ball_z = obs["ball"]
        ball_dx, ball_dy, ball_dz = obs["ball_direction"]
        # print(ball_z,ball_dz,ball_dx,ball_dy)
        ball_dx = self.sx(ball_dx)
        ball_dy = self.sy(ball_dy)
        self.ball_x, self.ball_y = self.tx(ball_x), self.ty(ball_y)
        ball_r = self.tr(ball_z)
        self.ball = self.canvas.create_oval(
            self.ball_x - ball_r,
            self.ball_y - ball_r,
            self.ball_x + ball_r,
            self.ball_y + ball_r,
            fill=self.ball_color,
        )
        self.buffers.append(self.ball)
        self.draw_direction(
            (ball_x, ball_y),
            (ball_dx, ball_dy),
            color=self.ball_color,
            scale=self.ball_direction_scale,
        )

    def draw_direction(self, pos, direction, color, scale):
        x, y = pos
        dx, dy = direction
        arrow = self.canvas.create_line(
            x, y, x + dx * scale, y + dy * scale, arrow=tk.LAST, dash=(3, 1), fill=color
        )
        self.buffers.append(arrow)

    def draw_ball_info(self, obs):
        x, y, z = obs["ball"]
        dx, dy, dz = obs["ball_direction"]
        rx, ry, rz = obs["ball_rotation"]
        s = "p:{:.2e},{:.2e},{:.2e}\nd:{:.2e},{:.2e},{:.2e}\nr:{:.2e},{:.2e},{:.2e}".format(
            x, y, z, dx, dy, dz, rx, ry, rz
        )
        # ix=max(50+self.x_off,min(self.pitch_width_frame-50+self.x_off,self.tx(x)+self.info_x_off))
        # iy=max(50+self.y_off,min(self.pitch_height_frame-50+self.y_off,self.ty(y)+self.info_y_off))
        ix = self.pitch_width_frame - 100
        iy = self.pitch_height_frame - 30
        ball_info = self.canvas.create_text(ix, iy, text=s, fill=self.info_color)
        self.buffers.append(ball_info)

    def clear_buffers(self):
        for item in self.buffers:
            self.canvas.delete(item)

    def create_players(self, obs):
        left_team_pos = obs["left_team"]
        right_team_pos = obs["right_team"]
        left_team_roles = list(map(SIMPLE_ROLES, obs["left_team_roles"]))
        right_team_roles = list(map(SIMPLE_ROLES, obs["right_team_roles"]))
        left_team_direciton = obs["left_team_direction"]
        right_team_direciton = obs["right_team_direction"]

        self.n_left = len(left_team_pos)
        self.n_right = len(right_team_pos)

        self.left_players = []
        self.right_players = []
        self.left_team_pos = []
        self.right_team_pos = []
        for i in range(self.n_left):
            x, y = left_team_pos[i]
            dx, dy = left_team_direciton[i]
            dx = self.sx(dx)
            dy = self.sy(dy)
            x = self.tx(x)
            y = self.ty(y)
            role = left_team_roles[i]
            player = self.canvas.create_text(
                x,
                y,
                text=role + str(i),
                fill=self.left_player_color,
                font=("Arial", 10, "bold"),
            )
            self.left_players.append(player)
            self.left_team_pos.append((x, y))
            self.draw_direction(
                (x, y), (dx, dy), self.left_player_color, self.player_direction_scale
            )
        for i in range(self.n_right):
            x, y = right_team_pos[i]
            dx, dy = right_team_direciton[i]
            dx = self.sx(dx)
            dy = self.sy(dy)
            x = self.tx(x)
            y = self.ty(y)
            role = right_team_roles[i]
            player = self.canvas.create_text(
                x,
                y,
                text=role + str(i),
                fill=self.right_player_color,
                font=("Arial", 10, "bold"),
            )
            self.right_players.append(player)
            self.right_team_pos.append((x, y))
            self.draw_direction(
                (x, y), (dx, dy), self.right_player_color, self.player_direction_scale
            )
        self.mark_ball_owner(obs)

    def sx(self, x):
        return x / self.pitch_width_coord * self.pitch_width_frame

    def sy(self, y):
        return y / self.pitch_height_coord * self.pitch_height_frame

    def mark_ball_owner(self, obs):
        ball_owned_team = obs["ball_owned_team"]
        ball_owned_player = obs["ball_owned_player"]
        if ball_owned_team == 0:
            players = self.left_players
            x, y = obs["left_team"][ball_owned_player]
            dx, dy = obs["left_team_direction"][ball_owned_player]
            dx = self.sx(dx) * 10
            dy = self.sy(dy) * 10
            x = self.tx(x)
            y = self.ty(y)
        elif ball_owned_team == 1:
            players = self.right_players
            x, y = obs["right_team"][ball_owned_player]
            dx, dy = obs["right_team_direction"][ball_owned_player]
            dx = self.sx(dx) * 10
            dy = self.sy(dy) * 10
            x = self.tx(x)
            y = self.ty(y)
        else:
            self.last_ball_owner = None
            return
        player = players[ball_owned_player]

        x, y = self.canvas.coords(player)
        circle = self.canvas.create_oval(
            x - self.mark_r,
            y - self.mark_r,
            x + self.mark_r,
            y + self.mark_r,
            outline=self.mark_color,
            width=2,
        )
        self.buffers.append(circle)

        self.draw_direction(
            (x, y), (dx, dy), self.mark_color, self.player_direction_scale
        )

    def update(self, obs):
        self.clear_buffers()
        self.update_ball(obs)
        self.update_players(obs)

    def create(self, obs):
        self.clear_buffers()
        self.create_pitch()
        self.create_ball(obs)
        self.create_players(obs)

    def update_ball(self, obs):
        self.draw_ball_info(obs)
        ball_x, ball_y, ball_z = obs["ball"]
        ball_dx, ball_dy, ball_dz = obs["ball_direction"]
        ball_x, ball_y = self.tx(ball_x), self.ty(ball_y)
        ball_dx = self.sx(ball_dx)
        ball_dy = self.sy(ball_dy)
        # print(ball_z,ball_dz,ball_dx,ball_dy)
        # self.canvas.move(self.ball,ball_x-self.ball_x,ball_y-self.ball_y)
        ball_r = self.tr(ball_z)
        self.ball = self.canvas.create_oval(
            ball_x - ball_r,
            ball_y - ball_r,
            ball_x + ball_r,
            ball_y + ball_r,
            fill=self.ball_color,
        )
        self.buffers.append(self.ball)
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.draw_direction(
            (ball_x, ball_y),
            (ball_dx, ball_dy),
            self.ball_color,
            self.ball_direction_scale,
        )

    def update_players(self, obs):
        left_team_pos = obs["left_team"]
        right_team_pos = obs["right_team"]
        left_team_direciton = obs["left_team_direction"]
        right_team_direciton = obs["right_team_direction"]

        new_left_team_pos = []
        new_right_team_pos = []
        for i in range(self.n_left):
            x, y = left_team_pos[i]
            dx, dy = left_team_direciton[i]
            dx = self.sx(dx)
            dy = self.sy(dy)
            x = self.tx(x)
            y = self.ty(y)
            old_x, old_y = self.left_team_pos[i]
            player = self.left_players[i]
            self.canvas.move(player, x - old_x, y - old_y)
            new_left_team_pos.append((x, y))
            self.draw_direction(
                (x, y), (dx, dy), self.left_player_color, self.player_direction_scale
            )
        self.left_team_pos = new_left_team_pos
        for i in range(self.n_right):
            x, y = right_team_pos[i]
            dx, dy = right_team_direciton[i]
            dx = self.sx(dx)
            dy = self.sy(dy)
            x = self.tx(x)
            y = self.ty(y)
            old_x, old_y = self.right_team_pos[i]
            player = self.right_players[i]
            self.canvas.move(player, x - old_x, y - old_y)
            new_right_team_pos.append((x, y))
            self.draw_direction(
                (x, y), (dx, dy), self.right_player_color, self.player_direction_scale
            )
        self.right_team_pos = new_right_team_pos
        self.mark_ball_owner(obs)

    def tx(self, x):
        return (
            x + self.pitch_width_coord / 2
        ) / self.pitch_width_coord * self.pitch_width_frame + self.x_off

    def ty(self, y):
        return (
            y + self.pitch_height_coord / 2
        ) / self.pitch_height_coord * self.pitch_height_frame + self.y_off

    def create_pitch(self):
        # todo use relative path
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radar.bmp")
        self.image = Image.open(path)
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(
            self.pitch_width_frame / 2 + self.x_off,
            self.pitch_height_frame / 2 + self.y_off,
            image=self.image,
        )
        self.canvas.create_rectangle(
            self.tx(-self.pitch_width_coord / 2),
            self.ty(-self.pitch_height_coord / 2),
            self.tx(self.pitch_width_coord / 2),
            self.ty(self.pitch_height_coord / 2),
            outline=self.pitch_color,
        )
        self.canvas.create_line(
            self.tx(0),
            self.ty(-self.pitch_height_coord / 2),
            self.tx(0),
            self.ty(self.pitch_height_coord / 2),
            fill=self.pitch_color,
        )
        self.canvas.create_rectangle(
            self.tx(-self.pitch_width_coord / 2) - self.door_d,
            self.ty(self.door_min_coord),
            self.tx(-self.pitch_width_coord / 2),
            self.ty(self.door_max_coord),
            outline=self.door_color,
            width=1.5,
        )
        self.canvas.create_rectangle(
            self.tx(self.pitch_width_coord / 2),
            self.ty(self.door_min_coord),
            self.tx(self.pitch_width_coord / 2) + self.door_d,
            self.ty(self.door_max_coord),
            outline=self.door_color,
            width=1.5,
        )

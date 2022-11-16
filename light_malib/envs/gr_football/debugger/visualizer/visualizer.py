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

from numpy import ndarray

import numpy as np
from PIL import Image, ImageTk
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter import ttk
import pandas as pd
from .translation_en import *
from .team_info_table import TeamInfoTable
from .raw_drawer import RawDrawer
from .ball_info_table import BallInfoTable
from tkinter.scrolledtext import ScrolledText
import pprint
import json
import tree
import io
import copy


class Visualizer:
    def __init__(self, tracer, disable_RGB=True, disable_reward=True) -> None:
        self.config = {"root": {"width": 1900, "height": 1000}}
        self.step = 0
        self.tracer = tracer
        self.disable_RGB = disable_RGB
        self.disable_reward = disable_reward

        self.data = self.tracer.data

        self.length = len(self.data)

        # load some extra data
        if hasattr(self.tracer, "extra_data") and "rewards" in self.tracer.extra_data:
            rewards = self.tracer.extra_data["rewards"]
            for idx, reward in enumerate(rewards):
                if idx < self.length:
                    self.data[idx]["rewards"] = reward
                else:
                    assert np.all(reward == 0)

        # print(self.data[0],len(self.data))

        self.fps = 24
        self.play_flag = False

        self.create()

        if not self.disable_RGB:
            for i in range(self.length):
                if i == 0 and "frame" not in self.data[i]:
                    self.disable_RGB = True
                    break
                try:
                    # print(i)
                    # print(self.data[i])
                    with io.BytesIO(self.tracer.data[i]["frame"]) as f:
                        image = Image.open(copy.deepcopy(f))
                    self.data[i]["frame"] = ImageTk.PhotoImage(
                        image.resize(
                            (
                                self.main_canvas.winfo_width(),
                                self.main_canvas.winfo_height(),
                            )
                        )
                    )
                except:
                    self.data[i]["frame"] = None
                    self.data[i].pop("frame")

        self.rgb_img_drawings = []
        self.stats_plots = []
        self.stats_img = None
        self.stats_drawing = False

        self.draw()
        self.left_info_table.xview_moveto(1)

    def draw_raw(self):
        self.raw_drawer.update(self.data[self.step])

    def draw_rgb(self):
        img = self.main_canvas.create_image(
            0, 0, anchor="nw", image=self.data[self.step]["frame"]
        )
        print(img, self.data[self.step]["frame"])
        if len(self.rgb_img_drawings) > 0:
            old_img = self.rgb_img_drawings.pop(0)
            self.main_canvas.delete(old_img)
        self.rgb_img_drawings.append(img)

    def draw_main(self):
        if not self.disable_RGB:
            print("draw rgb")
            self.draw_rgb()

    def draw_top_panel(self):
        self.draw_raw()

    def draw_top_left_panel(self):
        obs = self.data[self.step]
        self.left_info_table.update_table(obs, self.step)
        # self.left_info_table.xview_moveto(1)

    def draw_top_right_panel(self):
        obs = self.data[self.step]
        self.right_info_table.update_table(obs, self.step)

    def draw_left_panel(self):
        obs = self.data[self.step]
        obs = {k: v for k, v in obs.items() if k not in ["frame"]}
        obs = tree.map_structure(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x, obs
        )
        obs = tree.map_structure(
            lambda x: "{:.4}".format(x) if type(x) == float else x, obs
        )
        self.obs_info_text.delete("1.0", tk.END)
        self.obs_info_text.insert(tk.INSERT, pprint.pformat(obs))

    def draw_right_panel(self):
        obs = self.data[self.step]
        self.ball_info_table.update_table(obs, self.step)

    def draw(self):
        self.draw_main()
        self.draw_top_panel()
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_top_left_panel()
        self.draw_top_right_panel()
        self.draw_bottom_panel()

    def create(self):
        self.create_root_window()
        self.create_bottom_panel(self.root)
        self.create_main_panel(self.root)
        self.create_top_panel(self.root)
        self.create_left_panel(self.root)
        self.create_right_panel(self.root)
        self.create_top_left_panel(self.root)
        self.create_top_right_panel(self.root)
        self.root.update()
        self.raw_drawer = RawDrawer(self.top_canvas, self.data[0])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.update()

    def on_closing(self):
        self.play_flag = False
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.mainloop()

    def inc_step(self, n=1):
        self.step += n
        self.step = min(self.step, self.length - 1)
        self.update()

    def dec_step(self, n=1):
        self.step -= n
        self.step = max(self.step, 0)
        self.update()

    def slide_step(self, event):
        self.step = self.frame_slider.get()
        self.update()

    def update(self):
        self.frame_slider.set(self.step)
        self.left_step_label.config(
            text="left steps: {}".format(self.length - 1 - self.step)
        )
        self.draw()
        self.main_canvas.update()

    def draw_bottom_panel(self):
        obs = self.data[self.step]
        self.game_mode_label.config(
            text="game mode: {}".format(GAME_MODES(obs["game_mode"]))
        )
        self.score_label.config(text="score: {}:{}".format(*obs["score"]))

    def _play(self):
        if self.play_flag:
            self.step += 1
            if self.step == self.length:
                self.play_flag = False
                self.step = self.length - 1
            else:
                self.update()
                self.root.after(int(1 / self.fps * 1000), self._play)

    def play(self):
        if self.play_flag:
            self.play_flag = False
            return
        else:
            self.play_flag = True
            self._play()

    def stop(self):
        self.play_flag = False

    def set_fps(self):
        fps = self.fps_entry.get()
        try:
            fps = int(fps)
            if fps <= 0:
                raise Exception()
        except Exception as e:
            message = "Please input an int fps>0!"
            # print(message)
            showerror(message=message)
            return
        self.fps = fps
        self.fps_label.config(text="current fps: {}".format(self.fps))

    def create_bottom_panel(self, parent):
        self.main_button_group = tk.Frame(parent)
        self.main_button_group.place(relx=0.3, rely=0.83, relwidth=0.4, relheight=0.16)
        for i in range(4):
            self.main_button_group.grid_rowconfigure(
                i, weight=1, uniform="row_group_1"
            )  # this needed to be added？
        for i in range(6):
            self.main_button_group.grid_columnconfigure(
                i, weight=1, uniform="col_group_1"
            )  # as did this？
        self.frame_backward10 = tk.Button(
            self.main_button_group,
            text="-10 frame",
            command=partial(self.dec_step, n=10),
        )
        self.frame_backward10.grid(column=0, row=0, sticky="nesw", columnspan=1)
        self.frame_backward1 = tk.Button(
            self.main_button_group, text=" -1 frame", command=self.dec_step
        )
        self.frame_backward1.grid(column=1, row=0, sticky="nesw", columnspan=1)
        self.frame_backward1 = tk.Button(
            self.main_button_group, text="pause", command=self.stop
        )
        self.frame_backward1.grid(column=2, row=0, sticky="nesw", columnspan=1)
        self.frame_backward1 = tk.Button(
            self.main_button_group, text="play", command=self.play
        )
        self.frame_backward1.grid(column=3, row=0, sticky="nesw", columnspan=1)
        self.frame_forward1 = tk.Button(
            self.main_button_group, text=" +1 frame", command=self.inc_step
        )
        self.frame_forward1.grid(column=4, row=0, sticky="nesw", columnspan=1)
        self.frame_forward1 = tk.Button(
            self.main_button_group,
            text="+10 frame",
            command=partial(self.inc_step, n=10),
        )
        self.frame_forward1.grid(column=5, row=0, sticky="nesw", columnspan=1)
        self.frame_slider = tk.Scale(
            self.main_button_group,
            orient="horizontal",
            command=self.slide_step,
            from_=0,
            to=self.length - 1,
        )
        self.frame_slider.grid(column=0, row=1, sticky="nesw", columnspan=6)
        self.fps_label = tk.Label(
            self.main_button_group, text="current fps: {}".format(self.fps)
        )
        self.fps_label.grid(column=0, row=2, sticky="nesw", columnspan=1)
        self.fps_entry = tk.Entry(self.main_button_group)
        self.fps_entry.grid(column=1, row=2, sticky="nesw", columnspan=1)
        self.fps_setter = tk.Button(
            self.main_button_group, text="set fps", command=self.set_fps
        )
        self.fps_setter.grid(column=2, row=2, sticky="nesw", columnspan=1)
        self.left_step_label = tk.Label(
            self.main_button_group,
            text="left steps: {}".format(self.length - 1 - self.step),
        )
        self.left_step_label.grid(column=3, row=2, sticky="nesw", columnspan=1)
        self.total_step_label = tk.Label(
            self.main_button_group, text="total steps: {}".format(self.length - 1)
        )
        self.total_step_label.grid(column=4, row=2, sticky="nesw", columnspan=1)

        obs = self.data[0]
        self.game_mode_label = tk.Label(
            self.main_button_group,
            text="game mode: {}".format(GAME_MODES(obs["game_mode"])),
        )
        self.game_mode_label.grid(column=0, row=3, sticky="nesw", columnspan=1)
        self.score_label = tk.Label(
            self.main_button_group, text="score: {}:{}".format(*obs["score"])
        )
        self.score_label.grid(column=1, row=3, sticky="nesw", columnspan=1)

    def create_top_panel(self, parent):
        self.top_canvas = tk.Canvas(parent)
        self.top_canvas.place(relx=0.3, rely=0.01, relwidth=0.4, relheight=0.4)

    def create_main_panel(self, parent):
        self.main_canvas = tk.Canvas(parent)
        self.main_canvas.place(relx=0.3, rely=0.42, relwidth=0.4, relheight=0.4)

    def create_right_panel(self, parent):
        self.ball_info_table = BallInfoTable(parent, "left", self.data[0], 0)
        self.ball_info_table.place(relx=0.70, rely=0.22, relwidth=0.28, relheight=0.05)
        self.draw_right_panel()

    def create_left_panel(self, parent):
        self.obs_info_text = ScrolledText(parent)
        self.obs_info_text.place(relx=0.01, rely=0.22, relwidth=0.28, relheight=0.75)
        self.draw_left_panel()

    def create_top_right_panel(self, parent):
        self.right_info_table = TeamInfoTable(parent, "right", self.data[0], 0)
        self.right_info_table.place(relx=0.70, rely=0.05, relwidth=0.28, relheight=0.16)
        scrlbar = ttk.Scrollbar(
            self.right_info_table,
            orient="horizontal",
            command=self.right_info_table.xview,
        )
        self.right_info_table.configure(xscrollcommand=scrlbar.set)
        scrlbar.pack(side="bottom", fill="x")
        self.draw_top_right_panel()

    def create_top_left_panel(self, parent):
        # self.left_frame=tk.Frame(parent)
        # self.left_frame.place(relx=0.01,rely=0.01,relwidth=0.28,relheight=0.98)
        self.left_info_table = TeamInfoTable(parent, "left", self.data[0], 0)
        self.left_info_table.place(relx=0.01, rely=0.05, relwidth=0.28, relheight=0.16)
        self.left_info_table.xview_moveto(1)
        scrlbar = ttk.Scrollbar(
            self.left_info_table,
            orient="horizontal",
            command=self.left_info_table.xview,
        )
        self.left_info_table.configure(xscrollcommand=scrlbar.set)
        scrlbar.pack(side="bottom", fill="x")
        # verscrlbar.place(relx=0.0,rely=0.72,relwidth=.28,relheight=0.2)
        self.draw_top_left_panel()

    def create_root_window(self):
        self.root = tk.Tk()
        self.root.title("Google Research Football")

        window_width = self.config["root"]["width"]
        window_height = self.config["root"]["height"]

        # get the screen dimension
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        print("screen sizes(w,h)=({},{})".format(screen_width, screen_height))

        # find the center point
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.state("zoomed")

        style = ttk.Style()
        style.theme_use("clam")

        # # list the options of the style
        # # (Argument should be an element of TScrollbar, eg. "thumb", "trough", ...)
        # print(style.element_options("Horizontal.TScrollbar.thumb"))

        # configure the style
        # style.configure("Horizontal.TScrollbar", gripcount=2,
        #                 background="Green", darkcolor="DarkGreen", lightcolor="LightGreen",
        #                 troughcolor="gray", bordercolor="blue", arrowcolor="white",sliderlength=1)

    def mainloop(self):
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        finally:
            self.root.mainloop()


if __name__ == "__main__":
    fp = __file__
    folder, file = os.path.split(fp)
    v = Visualizer(os.path.join(folder, "ui_config.json"))
